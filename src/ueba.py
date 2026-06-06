import syslog
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
KNOWN_SERVERS = {'192.168.107.224', '192.168.107.225', '192.168.107.230', '192.168.107.235'}
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


def batch_dns_cv(df):
    dns = df[df['port'] == 53].sort_values(['src_ip', 'timestamp'])
    gaps = dns.groupby('src_ip')['timestamp'].diff()
    grouped = gaps.groupby(dns['src_ip'])
    means = grouped.mean()
    stds = grouped.std()
    return (stds / means).dropna()


def interval_cv(df, ip):
    d = df[df['src_ip'] == ip].sort_values('timestamp')
    gaps = d['timestamp'].diff().dropna()
    return gaps.std() / gaps.mean() if len(gaps) > 1 else np.nan


# ---- Rules ----

def rule_botnet(df_train, df_test):
    train_cv = {ip: interval_cv(df_train, ip) for ip in df_train['src_ip'].unique()}
    train_cv = {k: v for k, v in train_cv.items() if pd.notna(v)}
    flagged = set()
    details = {}
    for ip in df_test['src_ip'].unique():
        if ip not in train_cv:
            continue
        ip_data = df_test[df_test['src_ip'] == ip]
        internal = ip_data[ip_data['dst_ip'].apply(
            lambda x: str(x) not in KNOWN_SERVERS and ipaddress.IPv4Address(x) in PRIVATE_NET)]
        if len(internal) == 0:
            continue
        p2p_cv = interval_cv(internal, ip)
        if pd.notna(p2p_cv) and p2p_cv < train_cv[ip] * 0.5:
            flagged.add(ip)
            details[ip] = f'P2P to {sorted(internal["dst_ip"].unique())}, '
            details[ip] += f'p2p_cv={p2p_cv:.2f} < 0.5*train_cv={train_cv[ip]:.2f}'
    return flagged, details


def rule_exfil(int_test_stats, int_train_stats):
    train_dns = int_train_stats['dns_https_flow_ratio'].dropna()
    dns_threshold = int_test_stats['dns_https_flow_ratio'].dropna().median() * \
        (1 + 5 * (train_dns - train_dns.median()).abs().max() / train_dns.median())

    train_ratio = int_train_stats['https_up_down_ratio'].dropna()
    https_threshold = int_test_stats['https_up_down_ratio'].dropna().median() * \
        (1 + 5 * (train_ratio - train_ratio.median()).abs().max() / train_ratio.median())

    dns_flagged = set()
    https_flagged = set()
    details = {}
    for ip in int_test_stats.index:
        if ip not in int_train_stats.index:
            continue
        tst = int_test_stats.loc[ip]
        reasons = []
        if pd.notna(tst['dns_https_flow_ratio']) and tst['dns_https_flow_ratio'] > dns_threshold \
                and tst['dns_flows'] > DNS_FLOWS_P99:
            reasons.append(f'DNS: ratio={tst["dns_https_flow_ratio"]:.3f} flows={int(tst["dns_flows"])}')
            dns_flagged.add(ip)
        if pd.notna(tst['https_up_down_ratio']) and tst['https_up_down_ratio'] > https_threshold:
            reasons.append(f'HTTPS: ratio={tst["https_up_down_ratio"]:.4f}')
            https_flagged.add(ip)
        if reasons:
            details[ip] = reasons
    return dns_flagged, https_flagged, details


def rule_cc_dns(df_train, int_train_stats, df_test, int_test_stats):
    tr_dns_flows = int_train_stats['dns_flows']
    tr_p95 = tr_dns_flows.quantile(0.95)
    exfil_threshold = int_test_stats['dns_https_flow_ratio'].dropna().median() * \
        (1 + 5 * (int_train_stats['dns_https_flow_ratio'].dropna() -
         int_train_stats['dns_https_flow_ratio'].dropna().median()).abs().max() /
         int_train_stats['dns_https_flow_ratio'].dropna().median())
    train_cv = batch_dns_cv(df_train)
    test_cv = batch_dns_cv(df_test)
    flagged = set()
    details = {}
    for ip in sorted(set(tr_dns_flows.index) & set(int_test_stats.index)):
        trn = tr_dns_flows[ip]
        tst = int_test_stats.loc[ip, 'dns_flows']
        if tst < 100 or trn <= 0 or not (tst > tr_p95 and tst > trn * 3):
            continue
        if ip not in train_cv.index or ip not in test_cv.index:
            continue
        cv_ratio = test_cv[ip] / train_cv[ip] if train_cv[ip] > 0 else np.nan
        tst_ratio = int_test_stats.loc[ip, 'dns_https_flow_ratio']
        if pd.notna(cv_ratio) and cv_ratio < 0.5 and pd.notna(tst_ratio) and tst_ratio < exfil_threshold:
            flagged.add(ip)
            details[ip] = f'flows={int(tst)} train={int(trn)} cv_ratio={cv_ratio:.2f} ratio={tst_ratio:.3f}'
    return flagged, details


def rule_anomalous_dests(df_train, df_test, global_countries, reader_country):
    flow_thresholds = {}
    for src_ip in df_train['src_ip'].unique():
        ip_data = df_train[df_train['src_ip'] == src_ip]
        public = ip_data[~ip_data['dst_ip'].apply(lambda x: ipaddress.IPv4Address(x) in PRIVATE_NET)]
        flow_thresholds[src_ip] = public.groupby('dst_ip').size().median() if len(public) > 0 else 0
    flagged = set()
    details = {}
    for src_ip in df_test['src_ip'].unique():
        ip_data = df_test[df_test['src_ip'] == src_ip]
        public = ip_data[~ip_data['dst_ip'].apply(lambda x: ipaddress.IPv4Address(x) in PRIVATE_NET)]
        threshold = flow_thresholds.get(src_ip, 0)
        new = {}
        for dst in public['dst_ip'].unique():
            cc = get_country(dst, reader_country)
            if cc != 'XX' and cc not in global_countries:
                flow_count = len(public[public['dst_ip'] == dst])
                if flow_count > threshold:
                    if cc not in new:
                        new[cc] = []
                    new[cc].append(dst)
        if new:
            flagged.add(src_ip)
            details[src_ip] = '; '.join(f'{cc}: {sorted(dsts)}' for cc, dsts in sorted(new.items()))
    return flagged, details


def rule_external_users(df_train, df_test):
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
            details[ip] = f'off={test_off_pct:.2%} cv={test_cv:.2f}'
    return flagged, details


# ---- Report ----

def send_alerts(rule, flagged, details):
    syslog.openlog('ueba', logoption=syslog.LOG_PID)
    for ip in sorted(flagged):
        msg = f'{ip} [{rule}] {details[ip]}'
        syslog.syslog(syslog.LOG_ALERT, msg)


def print_rule(rule, flagged, details):
    print(f'\n--- {rule} ---')
    if flagged:
        print(f'  {len(flagged)} IPs:')
        for ip in sorted(flagged):
            print(f'    {ip}  |  {details[ip]}')
    else:
        print('  0 IPs')


# ---- Main ----

def main():
    df_int_train, df_int_test, df_ext_train, df_ext_test = load_data()
    print(f'Loaded: {len(df_int_train)} train flows, {len(df_int_test)} test flows')

    reader_country, reader_asn = load_geoip()
    int_train_stats = compute_internal_stats(df_int_train)
    int_test_stats = compute_internal_stats(df_int_test)

    per_ip_countries = build_per_ip_country_set(df_int_train, reader_country)
    global_countries = set()
    for c in per_ip_countries.values():
        global_countries.update(c)

    f_botnet, d_botnet = rule_botnet(df_int_train, df_int_test)
    f_exfil_dns, f_exfil_https, d_exfil = rule_exfil(int_test_stats, int_train_stats)
    f_cc, d_cc = rule_cc_dns(df_int_train, int_train_stats, df_int_test, int_test_stats)
    f_dests, d_dests = rule_anomalous_dests(df_int_train, df_int_test, global_countries, reader_country)
    f_ext, d_ext = rule_external_users(df_ext_train, df_ext_test)

    print_rule('BotNet', f_botnet, d_botnet)
    print_rule('Exfil (DNS+HTTPS)', f_exfil_dns | f_exfil_https, d_exfil)
    print_rule('C&C DNS', f_cc, d_cc)
    print_rule('Anomalous Destinations', f_dests, d_dests)
    print_rule('External Users', f_ext, d_ext)

    all_int = f_botnet | f_exfil_dns | f_exfil_https | f_cc | f_dests
    print(f'\n--- Blocklist ---')
    print(f'  Internal: {len(all_int)} IPs')
    for ip in sorted(all_int):
        rules = []
        if ip in f_botnet: rules.append('BotNet')
        if ip in f_exfil_dns: rules.append('Exfil(DNS)')
        if ip in f_exfil_https: rules.append('Exfil(HTTPS)')
        if ip in f_cc: rules.append('C&C')
        if ip in f_dests: rules.append('Dests')
        print(f'    {ip}  [{len(rules)}]  {", ".join(rules)}')
    print(f'  External: {len(f_ext)} IPs')
    for ip in sorted(f_ext):
        print(f'    {ip}')

    send_alerts('BotNet', f_botnet, d_botnet)
    send_alerts('Exfil', f_exfil_dns | f_exfil_https, d_exfil)
    send_alerts('C&C_DNS', f_cc, d_cc)
    send_alerts('AnomDest', f_dests, d_dests)
    send_alerts('ExtUser', f_ext, d_ext)

    reader_country.close()
    reader_asn.close()


if __name__ == '__main__':
    main()
