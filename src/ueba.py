import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ipaddress
import geoip2.database
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
GRAPHS_DIR = Path(__file__).resolve().parent / 'statistics' / 'graphs'

PRIVATE_NET = ipaddress.IPv4Network('192.168.0.0/16')

# Global thresholds derived from training data
DNS_HTTPS_RATIO_THRESHOLD = 0.20
DNS_MEAN_UP_THRESHOLD = 217
HTTPS_RATIO_THRESHOLD = 0.12


def load_data():
    internal_train = pd.read_json(DATA_DIR / 'internal_train7.json')
    internal_test = pd.read_json(DATA_DIR / 'internal_test7.json')
    external_train = pd.read_json(DATA_DIR / 'external_train7.json')
    external_test = pd.read_json(DATA_DIR / 'external_test7.json')
    return internal_train, internal_test, external_train, external_test


def load_geoip():
    reader_country = geoip2.database.Reader(str(DATA_DIR / 'dbip-country-lite-2026-05.mmdb'))
    reader_asn = geoip2.database.Reader(str(DATA_DIR / 'dbip-asn-lite-2026-05.mmdb'))
    return reader_country, reader_asn


def get_country(ip, reader):
    try:
        return reader.country(ip).country.iso_code
    except Exception:
        return 'XX'


def compute_internal_stats(df):
    df_sorted = df.sort_values(['src_ip', 'timestamp'])
    grouped = df_sorted.groupby('src_ip')

    dns = df[df['port'] == 53].groupby('src_ip')
    https = df[df['port'] == 443].groupby('src_ip')

    stats = pd.DataFrame(index=grouped.size().index)
    stats['total_flows'] = grouped.size()
    stats['distinct_dsts'] = grouped['dst_ip'].nunique()
    stats['dns_flows'] = dns.size().reindex(stats.index, fill_value=0)
    stats['dns_up'] = dns['up_bytes'].sum().reindex(stats.index, fill_value=0)
    stats['dns_mean_up'] = dns['up_bytes'].mean().reindex(stats.index, fill_value=0)
    stats['https_flows'] = https.size().reindex(stats.index, fill_value=0)
    stats['https_up'] = https['up_bytes'].sum().reindex(stats.index, fill_value=0)
    stats['https_down'] = https['down_bytes'].sum().reindex(stats.index, fill_value=0)
    stats['dns_https_flow_ratio'] = stats['dns_flows'] / stats['https_flows'].replace(0, np.nan)
    stats['https_up_down_ratio'] = stats['https_up'] / stats['https_down'].replace(0, np.nan)
    return stats


def build_per_ip_country_set(df, reader_country):
    per_ip = {}
    for src_ip in df['src_ip'].unique():
        ip_data = df[df['src_ip'] == src_ip]
        public = ip_data[~ip_data['dst_ip'].apply(lambda x: ipaddress.IPv4Address(x) in PRIVATE_NET)]
        countries = set()
        for dst in public['dst_ip'].unique():
            cc = get_country(dst, reader_country)
            if cc != 'XX':
                countries.add(cc)
        if countries:
            per_ip[src_ip] = countries
    return per_ip


def build_ext_baselines(df_ext):
    baselines = {}
    for src_ip in df_ext['src_ip'].unique():
        ip_data = df_ext[df_ext['src_ip'] == src_ip].sort_values('timestamp')
        intervals = ip_data['timestamp'].diff().dropna()
        hours = (ip_data['timestamp'] // 360000).unique()
        mean_int = intervals.mean() if len(intervals) > 0 else 0
        std_int = intervals.std() if len(intervals) > 0 else 0

        active_h = set(int(h) for h in hours)
        off = 0
        for h, c in (ip_data['timestamp'] // 360000).value_counts().items():
            if int(h) not in active_h:
                off += c
        max_offhours = off / len(ip_data) if len(ip_data) > 0 else 0

        baselines[src_ip] = {
            'flows': len(ip_data),
            'cv_interval': std_int / mean_int if mean_int > 0 else np.nan,
            'active_hours': active_h,
            'max_offhours': max_offhours,
        }
    return baselines


def check_per_ip_surge(test_val, train_val, min_abs, multiplier=3.0):
    if train_val <= 0:
        return False, test_val, train_val, 0
    ratio = test_val / train_val
    if ratio > multiplier and test_val > min_abs:
        return True, test_val, train_val, ratio
    return False, test_val, train_val, ratio


def rule_botnet(int_test_stats, int_train_stats):
    flagged = set()
    details = {}
    for ip in int_test_stats.index:
        if ip not in int_train_stats.index:
            continue
        trn = int_train_stats.loc[ip]
        tst = int_test_stats.loc[ip]

        hit_flows, vf, tf, rf = check_per_ip_surge(tst['total_flows'], trn['total_flows'], 2000)
        hit_dsts, vd, td, rd = check_per_ip_surge(tst['distinct_dsts'], trn['distinct_dsts'], 80)

        if hit_flows and hit_dsts:
            reasons = [
                f'flows {int(vf)} (train={int(tf)}, {rf:.1f}x)',
                f'dst IPs {int(vd)} (train={int(td)}, {rd:.1f}x)',
            ]
            flagged.add(ip)
            details[ip] = reasons
    return flagged, details


def rule_exfil_dns(int_test_stats, int_train_stats):
    flagged = set()
    details = {}
    for ip in int_test_stats.index:
        if ip not in int_train_stats.index:
            continue
        reasons = []
        trn = int_train_stats.loc[ip]
        tst = int_test_stats.loc[ip]

        ratio_hit = pd.notna(tst['dns_https_flow_ratio']) and tst['dns_https_flow_ratio'] > DNS_HTTPS_RATIO_THRESHOLD
        dns_surge, vf, tf, rf = check_per_ip_surge(tst['dns_flows'], trn['dns_flows'], 500, 5.0)

        if ratio_hit and dns_surge:
            reasons.append(f'DNS/HTTPS ratio {tst["dns_https_flow_ratio"]:.3f} > {DNS_HTTPS_RATIO_THRESHOLD}')
            reasons.append(f'DNS flows {int(vf)} (train={int(tf)}, {rf:.1f}x, 5x threshold)')

        up_surge, vu, tu, ru = check_per_ip_surge(tst['dns_up'], trn['dns_up'], 50000, 5.0)
        if up_surge:
            reasons.append(f'DNS upload {int(vu)} bytes (train={int(tu)}, {ru:.1f}x, 5x threshold)')

        if reasons:
            flagged.add(ip)
            details[ip] = reasons
    return flagged, details


def rule_exfil_https(int_test_stats, int_train_stats):
    flagged = set()
    details = {}
    for ip in int_test_stats.index:
        if ip not in int_train_stats.index:
            continue
        reasons = []
        trn = int_train_stats.loc[ip]
        tst = int_test_stats.loc[ip]

        hit, v, t, r = check_per_ip_surge(tst['https_up'], trn['https_up'], 20_000_000, 5.0)
        if hit:
            reasons.append(f'HTTPS upload {int(v)} bytes (train={int(t)}, {r:.1f}x, 5x threshold)')

        ratio_hit = pd.notna(tst['https_up_down_ratio']) and tst['https_up_down_ratio'] > HTTPS_RATIO_THRESHOLD
        vol_hit, _, _, _ = check_per_ip_surge(tst['https_up'], trn['https_up'], 20_000_000, 3.0)
        if ratio_hit and vol_hit:
            reasons.append(f'HTTPS up/down ratio {tst["https_up_down_ratio"]:.4f} > {HTTPS_RATIO_THRESHOLD} '
                           f'(plus volume surge)')

        if reasons:
            flagged.add(ip)
            details[ip] = reasons
    return flagged, details


def rule_cc_dns(int_test_stats, int_train_stats):
    flagged = set()
    details = {}
    for ip in int_test_stats.index:
        if ip not in int_train_stats.index:
            continue
        reasons = []
        trn = int_train_stats.loc[ip]
        tst = int_test_stats.loc[ip]

        hit_surge, vf, tf, rf = check_per_ip_surge(tst['dns_flows'], trn['dns_flows'], 100, 3.0)
        ratio_normal = pd.notna(tst['dns_https_flow_ratio']) and tst['dns_https_flow_ratio'] < 0.16
        large_packets = pd.notna(tst['dns_mean_up']) and tst['dns_mean_up'] > DNS_MEAN_UP_THRESHOLD

        if hit_surge and ratio_normal:
            reasons.append(f'DNS C&C beaconing: flows {int(vf)} (train={int(tf)}, {rf:.1f}x), '
                           f'normal ratio {tst["dns_https_flow_ratio"]:.3f}')
        if large_packets:
            reasons.append(f'DNS packet size {tst["dns_mean_up"]:.1f} bytes > {DNS_MEAN_UP_THRESHOLD}')
        if reasons:
            flagged.add(ip)
            details[ip] = reasons
    return flagged, details


def rule_anomalous_dests(df_test, global_countries_train, reader_country):
    flagged = set()
    details = {}
    for src_ip in df_test['src_ip'].unique():
        ip_data = df_test[df_test['src_ip'] == src_ip]
        public = ip_data[~ip_data['dst_ip'].apply(lambda x: ipaddress.IPv4Address(x) in PRIVATE_NET)]
        new = {}
        for dst in public['dst_ip'].unique():
            cc = get_country(dst, reader_country)
            if cc != 'XX' and cc not in global_countries_train:
                flow_count = len(public[public['dst_ip'] == dst])
                if flow_count >= 5:
                    if cc not in new:
                        new[cc] = []
                    new[cc].append(dst)
        if new:
            flagged.add(src_ip)
            details[src_ip] = []
            for cc in sorted(new):
                details[src_ip].append(f'{cc} ({", ".join(sorted(new[cc]))})')
    return flagged, details


def rule_external_users(df_test, ext_baselines):
    global_min_cv = min(b['cv_interval'] for b in ext_baselines.values()
                        if pd.notna(b['cv_interval']))
    print(f'  External user thresholds: global min CV = {global_min_cv:.2f}, '
          f'off-hours > 50% (AND with low CV) or > 80% (standalone)')
    print()

    flagged = set()
    details = {}
    for src_ip in df_test['src_ip'].unique():
        if src_ip not in ext_baselines:
            continue
        bl = ext_baselines[src_ip]
        ip_data = df_test[df_test['src_ip'] == src_ip].sort_values('timestamp')
        intervals = ip_data['timestamp'].diff().dropna()

        test_mean_int = intervals.mean() if len(intervals) > 0 else 0
        test_std_int = intervals.std() if len(intervals) > 0 else 0
        test_cv = test_std_int / test_mean_int if test_mean_int > 0 else np.nan

        hour_counts = (ip_data['timestamp'] // 360000).value_counts()
        total = len(ip_data)
        off = sum(c for h, c in hour_counts.items() if int(h) not in bl['active_hours'])
        off_pct = off / total if total > 0 else 0

        reasons = []
        has_temporal = off_pct > 0.50
        has_extreme = off_pct > 0.80
        has_regularity = pd.notna(test_cv) and test_cv < global_min_cv

        if has_temporal and has_regularity:
            reasons.append(f'temporal: {off_pct:.0%} off-hours')
            reasons.append(f'regularity: cv {test_cv:.2f} < global min cv={global_min_cv:.2f} '
                           f'(too periodic + off-hours)')
        elif has_extreme:
            reasons.append(f'extreme temporal shift: {off_pct:.0%} off-hours (80% threshold)')

        if reasons:
            flagged.add(src_ip)
            details[src_ip] = reasons

    return flagged, details


def print_results(rule_name, flagged, details, points):
    print(f'=== {rule_name} ({points} points) ===')
    if flagged:
        print(f'  Flagged IPs ({len(flagged)}):')
        for ip in sorted(flagged):
            print(f'    {ip}:')
            for reason in details.get(ip, []):
                print(f'      - {reason}')
    else:
        print('  No IPs flagged.')
    print()


def plot_test_vs_train(int_test_stats, int_train_stats, flagged_botnet, flagged_exfil_dns, flagged_dests):
    print('Generating anomaly comparison graphs...')
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    flagged_all = flagged_botnet | flagged_exfil_dns | flagged_dests
    merged = int_train_stats.join(int_test_stats, lsuffix='_train', rsuffix='_test', how='inner')

    train_common = merged.index[
        merged['https_flows_train'].notna() & merged['dns_flows_train'].notna()
    ]
    test_common = merged.index[
        merged['https_flows_test'].notna() & merged['dns_flows_test'].notna()
    ]
    common = train_common.intersection(test_common)
    if len(common) == 0:
        common = merged.index

    plt.figure(figsize=(10, 7))
    plt.scatter(merged.loc[common, 'https_flows_train'],
                merged.loc[common, 'dns_flows_train'],
                alpha=0.3, color='gray', label='Training', s=20)
    for ip in common:
        test_x = merged.loc[ip, 'https_flows_test']
        test_y = merged.loc[ip, 'dns_flows_test']
        if pd.notna(test_x) and pd.notna(test_y):
            color = 'red' if ip in flagged_all else 'steelblue'
            alpha = 0.9 if ip in flagged_all else 0.4
            marker = 'x' if ip in flagged_all else '.'
            size = 50 if ip in flagged_all else 15
            plt.plot([merged.loc[ip, 'https_flows_train'], test_x],
                     [merged.loc[ip, 'dns_flows_train'], test_y],
                     color='lightgray', linewidth=0.5, alpha=0.5)
            plt.scatter(test_x, test_y, alpha=alpha, color=color, s=size, marker=marker)
            if ip in flagged_all:
                plt.annotate(ip.split('.')[-1], (test_x, test_y), fontsize=7, color='darkred', fontweight='bold')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Training'),
        Line2D([0], [0], marker='.', color='w', markerfacecolor='steelblue', markersize=10, label='Test normal'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=10, label='Test anomalous'),
    ]
    plt.legend(handles=legend_elements)
    plt.xlabel('HTTPS flow count')
    plt.ylabel('DNS flow count')
    plt.title('DNS vs HTTPS — per-IP shift from training to test')
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'ueba_dns_vs_https.png')
    plt.close()
    print(f'  Saved {GRAPHS_DIR / "ueba_dns_vs_https.png"}')

    flow_ratio = merged['total_flows_test'] / merged['total_flows_train'].replace(0, np.nan)
    flow_ratio = flow_ratio.replace([np.inf, -np.inf], np.nan).dropna()
    botnet_mask = flow_ratio.index.isin(flagged_botnet)

    plt.figure(figsize=(10, 5))
    plt.hist(flow_ratio, bins=30, alpha=0.6, color='steelblue', label='All IPs', edgecolor='white')
    if botnet_mask.any():
        plt.hist(flow_ratio[botnet_mask], bins=30, alpha=0.8, color='red',
                 label='BotNet flagged', edgecolor='darkred')
    plt.axvline(x=3.0, color='red', linestyle='--', label='Threshold (3x)')
    plt.xlabel('Test/Training flow count ratio')
    plt.ylabel('Count')
    plt.title('Flow count deviation per IP — BotNet flagged IPs highlighted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'ueba_botnet_flows.png')
    plt.close()
    print(f'  Saved {GRAPHS_DIR / "ueba_botnet_flows.png"}')

    plt.figure(figsize=(10, 5))
    plt.hist(merged['dns_https_flow_ratio_test'].dropna(), bins=30, alpha=0.6,
             color='steelblue', label='Test IPs', edgecolor='white')
    exfil_mask = merged.index.isin(flagged_exfil_dns)
    if exfil_mask.any():
        plt.hist(merged.loc[exfil_mask, 'dns_https_flow_ratio_test'].dropna(), bins=30,
                 alpha=0.8, color='red', label='Exfil-DNS flagged', edgecolor='darkred')
    plt.axvline(x=DNS_HTTPS_RATIO_THRESHOLD, color='red', linestyle='--',
                label=f'Threshold ({DNS_HTTPS_RATIO_THRESHOLD})')
    plt.xlabel('DNS/HTTPS flow ratio')
    plt.ylabel('Count')
    plt.title('DNS/HTTPS ratio distribution — Exfil-DNS flagged IPs highlighted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'ueba_dns_ratio.png')
    plt.close()
    print(f'  Saved {GRAPHS_DIR / "ueba_dns_ratio.png"}')
    print()


def main():
    print('Loading data...')
    df_int_train, df_int_test, df_ext_train, df_ext_test = load_data()
    print(f'  internal_train: {len(df_int_train)} flows')
    print(f'  internal_test:  {len(df_int_test)} flows')
    print(f'  external_train: {len(df_ext_train)} flows')
    print(f'  external_test:  {len(df_ext_test)} flows')
    print()

    print('Loading GeoIP...')
    reader_country, reader_asn = load_geoip()
    print()

    print('Computing baselines from training...')
    int_train_stats = compute_internal_stats(df_int_train)
    int_test_stats = compute_internal_stats(df_int_test)
    per_ip_countries = build_per_ip_country_set(df_int_train, reader_country)
    global_countries = set()
    for c in per_ip_countries.values():
        global_countries.update(c)
    ext_baselines = build_ext_baselines(df_ext_train)
    print(f'  {len(int_train_stats)} training IPs, {len(int_test_stats)} test IPs')
    print(f'  {len(global_countries)} countries in global baseline')
    print(f'  {len(ext_baselines)} external IPs with baselines')
    print()

    common = set(int_train_stats.index) & set(int_test_stats.index)
    print(f'  {len(common)} IPs appear in both train and test')
    print()

    flagged_botnet, det_botnet = rule_botnet(int_test_stats, int_train_stats)
    flagged_exfil_dns, det_exfil_dns = rule_exfil_dns(int_test_stats, int_train_stats)
    flagged_exfil_https, det_exfil_https = rule_exfil_https(int_test_stats, int_train_stats)
    flagged_cc, det_cc = rule_cc_dns(int_test_stats, int_train_stats)
    flagged_dests, det_dests = rule_anomalous_dests(df_int_test, global_countries, reader_country)
    flagged_ext, det_ext = rule_external_users(df_ext_test, ext_baselines)

    print_results('Internal BotNet activity', flagged_botnet, det_botnet, 2)
    print_results('Data exfiltration via DNS', flagged_exfil_dns, det_exfil_dns, 4)
    print_results('Data exfiltration via HTTPS', flagged_exfil_https, det_exfil_https, 0)
    print_results('C&C via DNS', flagged_cc, det_cc, 2)
    print_results('Anomalous external destinations', flagged_dests, det_dests, 2)
    print_results('External user behavior', flagged_ext, det_ext, 2)

    plot_test_vs_train(int_test_stats, int_train_stats, flagged_botnet, flagged_exfil_dns, flagged_dests)

    all_internal_flagged = set()
    all_internal_flagged.update(flagged_botnet)
    all_internal_flagged.update(flagged_exfil_dns)
    all_internal_flagged.update(flagged_exfil_https)
    all_internal_flagged.update(flagged_cc)
    all_internal_flagged.update(flagged_dests)

    print('=' * 60)
    print('SUMMARY TABLE')
    print('=' * 60)
    print(f'  {"Rule":<40} {"Count":>6}  {"Points":>6}')
    print(f'  {"-"*40} {"-"*6}  {"-"*6}')
    print(f'  {"Internal BotNet activity":<40} {len(flagged_botnet):>6}  {"2":>6}')
    print(f'  {"Data exfiltration via DNS":<40} {len(flagged_exfil_dns):>6}  {"4":>6}')
    print(f'  {"Data exfiltration via HTTPS":<40} {len(flagged_exfil_https):>6}  {"-":>6}')
    print(f'  {"C&C via DNS":<40} {len(flagged_cc):>6}  {"2":>6}')
    print(f'  {"Anomalous external destinations":<40} {len(flagged_dests):>6}  {"2":>6}')
    print(f'  {"External user behaviour":<40} {len(flagged_ext):>6}  {"2":>6}')
    print(f'  {"-"*40} {"-"*6}  {"-"*6}')
    print(f'  {"Total unique internal IPs":<40} {len(all_internal_flagged):>6}')
    print(f'  {"Total unique external IPs":<40} {len(flagged_ext):>6}')
    print()

    print('=' * 60)
    print('DEVICES TO BLOCK')
    print('=' * 60)
    if all_internal_flagged:
        print(f'  {len(all_internal_flagged)} internal IPs (any detected anomaly):')
        for ip in sorted(all_internal_flagged):
            rules = []
            if ip in flagged_botnet:
                rules.append('BotNet')
            if ip in flagged_exfil_dns:
                rules.append('Exfil-DNS')
            if ip in flagged_exfil_https:
                rules.append('Exfil-HTTPS')
            if ip in flagged_cc:
                rules.append('C&C-DNS')
            if ip in flagged_dests:
                rules.append('Anomalous-dest')
            n = len(rules)
            conf = 'HIGH' if n >= 3 else ('MEDIUM' if n >= 2 else 'LOW')
            print(f'    {ip}  [{conf}, {n} rules]  {", ".join(rules)}')
    else:
        print('  No internal IPs to block.')
    if flagged_ext:
        print(f'  {len(flagged_ext)} external IPs:')
        for ip in sorted(flagged_ext):
            print(f'    {ip}')
    else:
        print('  No external IPs to block.')

    reader_country.close()
    reader_asn.close()
    print()
    print('Done.')


if __name__ == '__main__':
    main()
