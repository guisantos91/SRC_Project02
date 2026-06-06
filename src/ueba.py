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

DNS_MEAN_UP_THRESHOLD = 217
DNS_FLOWS_P99 = 1131


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


def check_per_ip_surge(test_val, train_val, min_abs, multiplier=3.0):
    if train_val <= 0:
        return False, test_val, train_val, 0
    ratio = test_val / train_val
    if ratio > multiplier and test_val > min_abs:
        return True, test_val, train_val, ratio
    return False, test_val, train_val, ratio


def batch_dns_cv(df):
    dns = df[df['port'] == 53].sort_values(['src_ip', 'timestamp'])
    gaps = dns.groupby('src_ip')['timestamp'].diff()
    grouped = gaps.groupby(dns['src_ip'])
    means = grouped.mean()
    stds = grouped.std()
    return (stds / means).dropna()


# ---- Rules ----

def rule_botnet(int_test_stats, int_train_stats, df_int_test, df_int_train):
    KNOWN_SERVERS = {'192.168.107.224', '192.168.107.225', '192.168.107.230', '192.168.107.235'}

    # Per-IP training CV baselines
    train_cv = {}
    for ip in df_int_train['src_ip'].unique():
        ip_d = df_int_train[df_int_train['src_ip'] == ip].sort_values('timestamp')
        gaps = ip_d['timestamp'].diff().dropna()
        cv = gaps.std() / gaps.mean() if len(gaps) > 1 else np.nan
        if pd.notna(cv):
            train_cv[ip] = cv

    flagged = set()
    details = {}
    for ip in int_test_stats.index:
        if ip not in int_train_stats.index or ip not in train_cv:
            continue

        ip_data = df_int_test[df_int_test['src_ip'] == ip]
        internal = ip_data[ip_data['dst_ip'].apply(lambda x: str(x) not in KNOWN_SERVERS
                                                     and ipaddress.IPv4Address(x) in PRIVATE_NET)]
        if len(internal) == 0:
            continue

        p2p_dsts = sorted(internal['dst_ip'].unique())
        p2p_flows = len(internal)
        p2p_sorted = internal.sort_values('timestamp')
        gaps = p2p_sorted['timestamp'].diff().dropna()
        p2p_cv = gaps.std() / gaps.mean() if len(gaps) > 1 else np.nan

        if pd.notna(p2p_cv) and p2p_cv < train_cv[ip] * 0.5:
            flagged.add(ip)
            details[ip] = [
                f'internal P2P: {p2p_flows} flows to {p2p_dsts}',
                f'  P2P CV={p2p_cv:.2f} < 0.5x train CV={train_cv[ip]:.2f} '
                f'({p2p_cv/train_cv[ip]:.2f}x — automated)',
            ]
    return flagged, details


def rule_exfil(int_test_stats, int_train_stats):
    train_dns = int_train_stats['dns_https_flow_ratio'].dropna()
    dns_tr_med = train_dns.median()
    dns_max_pct = (train_dns - dns_tr_med).abs().max() / dns_tr_med
    test_dns = int_test_stats['dns_https_flow_ratio'].dropna()
    dns_ts_med = test_dns.median()
    dns_threshold = dns_ts_med * (1 + 5 * dns_max_pct)

    train_ratio = int_train_stats['https_up_down_ratio'].dropna()
    ratio_median = train_ratio.median()
    max_pct = (train_ratio - ratio_median).abs().max() / ratio_median
    test_ratio = int_test_stats['https_up_down_ratio'].dropna()
    test_median = test_ratio.median()
    https_threshold = test_median * (1 + 5 * max_pct)

    print(f'  DNS exfil: train max rel dev from med={dns_max_pct*100:.2f}%, test median={dns_ts_med:.4f}, threshold={dns_threshold:.4f}')
    print(f'  HTTPS exfil: train max rel dev from med={max_pct*100:.2f}%, test median={test_median:.4f}, threshold={https_threshold:.4f}')
    print()

    dns_flagged = set()
    https_flagged = set()
    details = {}
    for ip in int_test_stats.index:
        if ip not in int_train_stats.index:
            continue
        tst = int_test_stats.loc[ip]
        dns_hit = pd.notna(tst['dns_https_flow_ratio']) and tst['dns_https_flow_ratio'] > dns_threshold
        dns_volume = tst['dns_flows'] > DNS_FLOWS_P99
        https_hit = pd.notna(tst['https_up_down_ratio']) and tst['https_up_down_ratio'] > https_threshold
        if (dns_hit and dns_volume) or https_hit:
            reasons = []
            if dns_hit and dns_volume:
                reasons.append(f'DNS exfil: ratio {tst["dns_https_flow_ratio"]:.3f} > {dns_threshold:.4f}, flows {int(tst["dns_flows"])} > P99({DNS_FLOWS_P99})')
                dns_flagged.add(ip)
            if https_hit:
                reasons.append(f'HTTPS exfil: ratio {tst["https_up_down_ratio"]:.4f} > {https_threshold:.4f}')
                https_flagged.add(ip)
            details[ip] = reasons
    return dns_flagged, https_flagged, details


def rule_cc_dns(df_int_test, int_test_stats, df_int_train, int_train_stats):
    tr_dns_flows = int_train_stats['dns_flows']
    tr_p95 = tr_dns_flows.quantile(0.95)
    exfil_dns = int_test_stats['dns_https_flow_ratio'].dropna()
    exfil_threshold = exfil_dns.median() * (1 + 5 * (
        (int_train_stats['dns_https_flow_ratio'].dropna() - int_train_stats['dns_https_flow_ratio'].dropna().median()).abs().max()
        / int_train_stats['dns_https_flow_ratio'].dropna().median()))
    common = set(tr_dns_flows.index) & set(int_test_stats.index)
    train_cv = batch_dns_cv(df_int_train)
    test_cv = batch_dns_cv(df_int_test)
    flagged = set()
    details = {}
    for ip in sorted(common):
        trn = tr_dns_flows[ip]
        tst = int_test_stats.loc[ip, 'dns_flows']
        if tst < 100 or trn <= 0 or not (tst > tr_p95 and tst > trn * 3):
            continue
        if ip not in train_cv.index or ip not in test_cv.index:
            continue
        trn_cv = train_cv[ip]
        tst_cv = test_cv[ip]
        cv_ratio = tst_cv / trn_cv if trn_cv > 0 else np.nan
        tst_ratio = int_test_stats.loc[ip, 'dns_https_flow_ratio']
        if pd.notna(cv_ratio) and cv_ratio < 0.5 and pd.notna(tst_ratio) and tst_ratio < exfil_threshold:
            flagged.add(ip)
            details[ip] = [
                f'beaconing: flows {int(tst)} (train={int(trn)}, {tst/trn:.1f}x)',
                f'cv {trn_cv:.2f}->{tst_cv:.2f} ({cv_ratio:.2f}x periodic)',
                f'ratio {tst_ratio:.3f} < exfil threshold {exfil_threshold:.3f}',
            ]
    print(f'  C&C-DNS: P95 train dns flows={tr_p95:.0f}, exfil threshold={exfil_threshold:.4f}')
    print(f'    beaconing: flows > P95 AND > 3x train AND cv_ratio < 0.5 AND ratio < exfil')
    print()
    return flagged, details


def rule_anomalous_dests(df_test, global_countries_train, reader_country, df_train):
    flow_thresholds = {}
    for src_ip in df_train['src_ip'].unique():
        ip_data = df_train[df_train['src_ip'] == src_ip]
        public = ip_data[~ip_data['dst_ip'].apply(lambda x: ipaddress.IPv4Address(x) in PRIVATE_NET)]
        if len(public) == 0:
            flow_thresholds[src_ip] = 0
            continue
        flow_thresholds[src_ip] = public.groupby('dst_ip').size().median()
    flagged = set()
    details = {}
    for src_ip in df_test['src_ip'].unique():
        ip_data = df_test[df_test['src_ip'] == src_ip]
        public = ip_data[~ip_data['dst_ip'].apply(lambda x: ipaddress.IPv4Address(x) in PRIVATE_NET)]
        threshold = flow_thresholds.get(src_ip, 0)
        new = {}
        for dst in public['dst_ip'].unique():
            cc = get_country(dst, reader_country)
            if cc != 'XX' and cc not in global_countries_train:
                flow_count = len(public[public['dst_ip'] == dst])
                if flow_count > threshold:
                    if cc not in new:
                        new[cc] = []
                    new[cc].append(dst)
        if new:
            flagged.add(src_ip)
            details[src_ip] = [f'{cc} ({", ".join(sorted(dsts))})' for cc, dsts in sorted(new.items())]
    return flagged, details


def rule_external_users(df_test, df_train):
    # Training baselines
    train_data = []
    for ip in df_train['src_ip'].unique():
        ip_d = df_train[df_train['src_ip'] == ip].sort_values('timestamp')
        gaps = ip_d['timestamp'].diff().dropna()
        cv = gaps.std() / gaps.mean() if len(gaps) > 1 else np.nan
        off = len(ip_d[(ip_d['timestamp'] // 360000).between(0, 5)])
        off_pct = off / len(ip_d) if len(ip_d) > 0 else 0
        train_data.append({'cv': cv, 'off_pct': off_pct, 'ip': ip})
    train_df = pd.DataFrame(train_data).dropna(subset=['cv'])
    p10_cv = train_df['cv'].quantile(0.10)
    p95_off = train_df['off_pct'].quantile(0.95)

    print(f'  External user thresholds:')
    print(f'    P10(train CV) = {p10_cv:.2f}  (more periodic than 90% of users)')
    print(f'    P95(train off-hours %) = {p95_off:.2%}  (more nighttime traffic than 95% of users)')
    print(f'    AND logic: both must fire')
    print()

    flagged = set()
    details = {}
    for ip in df_test['src_ip'].unique():
        ip_d = df_test[df_test['src_ip'] == ip].sort_values('timestamp')
        gaps = ip_d['timestamp'].diff().dropna()
        test_cv = gaps.std() / gaps.mean() if len(gaps) > 1 else np.nan
        off = len(ip_d[(ip_d['timestamp'] // 360000).between(0, 5)])
        test_off_pct = off / len(ip_d) if len(ip_d) > 0 else 0
        if pd.notna(test_cv) and test_cv < p10_cv and test_off_pct > p95_off:
            flagged.add(ip)
            details[ip] = [
                f'off-hours: {test_off_pct:.2%} > P95 train ({p95_off:.2%})',
                f'cv: {test_cv:.2f} < P10 train ({p10_cv:.2f})',
            ]
    return flagged, details


# ---- Output ----

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


def plot_test_vs_train(int_test_stats, int_train_stats, df_int_train, df_int_test,
                       flagged_botnet, flagged_exfil_dns, flagged_exfil_https,
                       flagged_cc, flagged_dests):
    print('Generating graphs...')
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    merged = int_train_stats.join(int_test_stats, lsuffix='_train', rsuffix='_test', how='inner')
    flagged_all = flagged_botnet | flagged_exfil_dns | flagged_dests

    # DNS vs HTTPS scatter
    common = merged.index.intersection(merged.index)
    plt.figure(figsize=(10, 7))
    plt.scatter(merged.loc[common, 'https_flows_train'], merged.loc[common, 'dns_flows_train'],
                alpha=0.3, color='gray', label='Training', s=20)
    for ip in common:
        tx, ty = merged.loc[ip, 'https_flows_test'], merged.loc[ip, 'dns_flows_test']
        if pd.notna(tx) and pd.notna(ty):
            c = 'red' if ip in flagged_all else 'steelblue'
            a = 0.9 if ip in flagged_all else 0.4
            m = 'x' if ip in flagged_all else '.'
            s = 50 if ip in flagged_all else 15
            plt.plot([merged.loc[ip, 'https_flows_train'], tx], [merged.loc[ip, 'dns_flows_train'], ty],
                     color='lightgray', linewidth=0.5, alpha=0.5)
            plt.scatter(tx, ty, alpha=a, color=c, s=s, marker=m)
            if ip in flagged_all:
                plt.annotate(ip.split('.')[-1], (tx, ty), fontsize=7, color='darkred', fontweight='bold')
    from matplotlib.lines import Line2D
    plt.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Training'),
        Line2D([0], [0], marker='.', color='w', markerfacecolor='steelblue', markersize=10, label='Test normal'),
        Line2D([0], [0], marker='x', color='w', markerfacecolor='red', markersize=10, label='Test anomalous'),
    ])
    plt.xlabel('HTTPS flow count')
    plt.ylabel('DNS flow count')
    plt.title('DNS vs HTTPS — per-IP shift from training to test')
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'ueba_dns_vs_https.png')
    plt.close()
    print(f'  Saved {GRAPHS_DIR / "ueba_dns_vs_https.png"}')

    # BotNet flow ratio histogram
    fr = (merged['total_flows_test'] / merged['total_flows_train'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(10, 5))
    plt.hist(fr, bins=30, alpha=0.6, color='steelblue', label='All IPs', edgecolor='white')
    bm = fr.index.isin(flagged_botnet)
    if bm.any():
        plt.hist(fr[bm], bins=30, alpha=0.8, color='red', label='BotNet flagged', edgecolor='darkred')
    plt.axvline(x=3.0, color='red', linestyle='--', label='Threshold (3x)')
    plt.xlabel('Test/Training flow count ratio')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'ueba_botnet_flows.png')
    plt.close()
    print(f'  Saved {GRAPHS_DIR / "ueba_botnet_flows.png"}')

    # DNS/HTTPS ratio histogram
    train_dns_r = int_train_stats['dns_https_flow_ratio'].dropna()
    dns_tr_med = train_dns_r.median()
    dns_max_p = (train_dns_r - dns_tr_med).abs().max() / dns_tr_med
    test_dns_r = int_test_stats['dns_https_flow_ratio'].dropna()
    dns_ts_med = test_dns_r.median()
    dns_t = dns_ts_med * (1 + 5 * dns_max_p)
    plt.figure(figsize=(10, 5))
    plt.hist(merged['dns_https_flow_ratio_test'].dropna(), bins=30, alpha=0.6, color='steelblue', label='Test IPs', edgecolor='white')
    em = merged.index.isin(flagged_exfil_dns)
    if em.any():
        plt.hist(merged.loc[em, 'dns_https_flow_ratio_test'].dropna(), bins=30, alpha=0.8, color='red', label='Exfil-DNS flagged', edgecolor='darkred')
    plt.axvline(x=dns_t, color='red', linestyle='--', label=f'Threshold ({dns_t:.4f})')
    plt.xlabel('DNS/HTTPS flow ratio')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'ueba_dns_ratio.png')
    plt.close()
    print(f'  Saved {GRAPHS_DIR / "ueba_dns_ratio.png"}')

    # HTTPS ratio compare
    tr_s = int_train_stats['https_up_down_ratio'].dropna()
    ts_s = int_test_stats['https_up_down_ratio'].dropna()
    tr_med = tr_s.median()
    tr_mp = (tr_s - tr_med).abs().max() / tr_med
    ts_med = ts_s.median()
    t = ts_med * (1 + 5 * tr_mp)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(tr_s, bins=30, edgecolor='black', alpha=0.8, color='gray')
    ax1.axvline(x=tr_med, color='steelblue', label=f'Median={tr_med:.4f}')
    ax1.axvline(x=tr_med*(1+tr_mp), color='orange', linestyle='--', label=f'±{tr_mp*100:.2f}%')
    ax1.axvline(x=tr_med*(1-tr_mp), color='orange', linestyle='--')
    ax1.set_title('Training — HTTPS up/down ratio')
    ax1.legend(fontsize=8)
    ax2.hist(ts_s, bins=30, edgecolor='black', alpha=0.8, color='steelblue')
    hm = ts_s.index.isin(flagged_exfil_https)
    if hm.any():
        ax2.hist(ts_s[hm], bins=30, edgecolor='darkred', alpha=0.9, color='red', label='HTTPS exfil flagged')
    ax2.axvline(x=ts_med, color='gray', label=f'Median={ts_med:.4f}')
    ax2.axvline(x=t, color='red', linestyle='--', label=f'Threshold={t:.4f}')
    ax2.set_title(f'Test — HTTPS up/down ratio')
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'ueba_https_ratio_compare.png')
    plt.close()
    print(f'  Saved {GRAPHS_DIR / "ueba_https_ratio_compare.png"}')

    # C&C DNS beaconing scatter
    tf = int_train_stats['dns_flows'].dropna()
    tr_p95 = tf.quantile(0.95)
    tc = batch_dns_cv(df_int_test)
    rc = batch_dns_cv(df_int_train)
    pts = []
    for ip in sorted(set(rc.index) & set(tc.index) & set(tf.index) & set(int_test_stats.index)):
        trn = tf[ip]
        tst = int_test_stats.loc[ip, 'dns_flows']
        if trn <= 0 or tst < 100:
            continue
        pts.append({'ip': ip, 'flow_ratio': tst/trn, 'cv_ratio': tc[ip]/rc[ip],
                     'tst_flows': tst, 'trn_flows': trn, 'flagged': ip in flagged_cc})
    if pts:
        dp = pd.DataFrame(pts)
        n = dp[~dp['flagged']]
        f = dp[dp['flagged']]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.scatter(n['flow_ratio'], n['cv_ratio'], alpha=0.4, color='steelblue', s=15, label='Normal')
        if len(f) > 0:
            ax1.scatter(f['flow_ratio'], f['cv_ratio'], alpha=0.9, color='red', s=50, marker='x', edgecolors='darkred', label='C&C flagged')
            for _, r in f.iterrows():
                ax1.annotate(r['ip'].split('.')[-1], (r['flow_ratio'], r['cv_ratio']), fontsize=7, color='darkred')
        ax1.axhline(y=0.5, color='red', linestyle='--', label='cv_ratio < 0.5')
        ax1.axvline(x=3.0, color='orange', linestyle='--', label='flow > 3x')
        ax1.set_xlabel('DNS flow ratio (test/train)')
        ax1.set_ylabel('CV ratio (test/train)')
        ax1.set_title('DNS beaconing — CV vs flow surge')
        ax1.legend(fontsize=8)
        ax2.scatter(n['tst_flows'], n['cv_ratio'], alpha=0.4, color='steelblue', s=15)
        if len(f) > 0:
            ax2.scatter(f['tst_flows'], f['cv_ratio'], alpha=0.9, color='red', s=50, marker='x', edgecolors='darkred')
            for _, r in f.iterrows():
                ax2.annotate(r['ip'].split('.')[-1], (r['tst_flows'], r['cv_ratio']), fontsize=7, color='darkred')
        ax2.axhline(y=0.5, color='red', linestyle='--', label='cv_ratio < 0.5')
        ax2.axvline(x=tr_p95, color='orange', linestyle='--', label=f'P95 flow={tr_p95:.0f}')
        ax2.set_xlabel('DNS flow count (test)')
        ax2.set_ylabel('CV ratio (test/train)')
        ax2.set_title('DNS beaconing — CV vs flow volume')
        ax2.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(GRAPHS_DIR / 'ueba_cc_dns_beaconing.png')
        plt.close()
        print(f'  Saved {GRAPHS_DIR / "ueba_cc_dns_beaconing.png"}')
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
    print(f'  {len(int_train_stats)} training IPs, {len(int_test_stats)} test IPs')
    print(f'  {len(global_countries)} countries in global baseline')
    print()

    flagged_botnet, det_botnet = rule_botnet(int_test_stats, int_train_stats, df_int_test, df_int_train)
    flagged_exfil_dns, flagged_exfil_https, det_exfil = rule_exfil(int_test_stats, int_train_stats)
    flagged_cc, det_cc = rule_cc_dns(df_int_test, int_test_stats, df_int_train, int_train_stats)
    flagged_dests, det_dests = rule_anomalous_dests(df_int_test, global_countries, reader_country, df_int_train)
    flagged_ext, det_ext = rule_external_users(df_ext_test, df_ext_train)

    print_results('Internal BotNet activity', flagged_botnet, det_botnet, 2)
    print_results('Data exfiltration (DNS + HTTPS)', flagged_exfil_dns | flagged_exfil_https, det_exfil, 4)
    print_results('C&C via DNS', flagged_cc, det_cc, 2)
    print_results('Anomalous external destinations', flagged_dests, det_dests, 2)
    print_results('External user behavior', flagged_ext, det_ext, 2)

    plot_test_vs_train(int_test_stats, int_train_stats, df_int_train, df_int_test,
                       flagged_botnet, flagged_exfil_dns, flagged_exfil_https,
                       flagged_cc, flagged_dests)

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
    print(f'  {"Data exfil (DNS + HTTPS)":<40} {len(flagged_exfil_dns | flagged_exfil_https):>6}  {"4":>6}')
    print(f'  {"  of which DNS":<40} {len(flagged_exfil_dns):>6}')
    print(f'  {"  of which HTTPS":<40} {len(flagged_exfil_https):>6}')
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
        print(f'  {len(all_internal_flagged)} internal IPs:')
        for ip in sorted(all_internal_flagged):
            rules = []
            if ip in flagged_botnet: rules.append('BotNet')
            if ip in flagged_exfil_dns: rules.append('Exfil(DNS)')
            if ip in flagged_exfil_https: rules.append('Exfil(HTTPS)')
            if ip in flagged_cc: rules.append('C&C-DNS')
            if ip in flagged_dests: rules.append('Anomalous-dest')
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
