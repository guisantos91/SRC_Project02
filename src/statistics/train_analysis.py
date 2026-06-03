import pandas as pd
import numpy as np
import ipaddress
import geoip2.database
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path('../../data')
GRAPHS_DIR = Path('graphs')
GRAPHS_DIR.mkdir(exist_ok=True)


def load_data():
    internal = pd.read_json(DATA_DIR / 'internal_train7.json')
    external = pd.read_json(DATA_DIR / 'external_train7.json')
    return internal, external


def load_geoip():
    reader_country = geoip2.database.Reader(str(DATA_DIR / 'dbip-country-lite-2026-05.mmdb'))
    reader_asn = geoip2.database.Reader(str(DATA_DIR / 'dbip-asn-lite-2026-05.mmdb'))
    return reader_country, reader_asn


def get_country(ip, reader):
    try:
        return reader.country(ip).country.iso_code
    except Exception:
        return 'XX'


def get_asn(ip, reader):
    try:
        return reader.asn(ip).autonomous_system_number
    except Exception:
        return -1


def get_asn_org(ip, reader):
    try:
        return reader.asn(ip).autonomous_system_organization
    except Exception:
        return 'Unknown'


def identify_private_network(df_internal):
    print('=' * 60)
    print('1. PRIVATE NETWORK IDENTIFICATION')
    print('=' * 60)

    unique_ips = sorted(df_internal['src_ip'].unique())
    print(f'Total unique internal IPs: {len(unique_ips)}')

    private_ranges = []
    for ip_str in unique_ips:
        ip = ipaddress.IPv4Address(ip_str)
        for net in [ipaddress.IPv4Network('10.0.0.0/8'),
                     ipaddress.IPv4Network('172.16.0.0/12'),
                     ipaddress.IPv4Network('192.168.0.0/16')]:
            if ip in net:
                private_ranges.append(net)
                break

    private_ranges = list(set(private_ranges))
    print(f'Private network range(s) identified:')
    for r in private_ranges:
        print(f'  {r}')
    print(f'Internal IP list:')
    for ip in unique_ips:
        print(f'  {ip}')
    print()
    return private_ranges[0] if private_ranges else None


def identify_internal_servers(df_internal, private_network):
    print('=' * 60)
    print('2. INTERNAL SERVERS / SERVICES')
    print('=' * 60)

    mask = df_internal['dst_ip'].apply(lambda x: ipaddress.IPv4Address(x) in private_network)
    internal_flows = df_internal[mask]

    if len(internal_flows) == 0:
        print('No internal-to-internal flows found.')
        print()
        return

    server_stats = internal_flows.groupby('dst_ip').agg(
        flow_count=('src_ip', 'count'),
        distinct_sources=('src_ip', 'nunique'),
        up_bytes=('up_bytes', 'sum'),
        down_bytes=('down_bytes', 'sum'),
        ports=('port', lambda x: sorted(x.unique())),
        protos=('proto', lambda x: sorted(x.unique()))
    ).sort_values('flow_count', ascending=False)

    print(f'Internal servers (IPs that receive internal traffic):')
    for ip, row in server_stats.iterrows():
        print(f'  {ip}: {int(row["flow_count"])} flows, '
              f'{int(row["distinct_sources"])} sources, '
              f'up={int(row["up_bytes"])} down={int(row["down_bytes"])}, '
              f'ports={row["ports"]}, proto={row["protos"]}')
    print()
    return server_stats


def analyze_src_ip_stats(df_internal, reader_country, reader_asn):
    print('=' * 60)
    print('3. INTERNAL USERS TRAFFIC - PER SOURCE IP')
    print('=' * 60)

    grouped = df_internal.groupby('src_ip')

    total_flows = grouped.size()
    total_up = grouped['up_bytes'].sum()
    total_down = grouped['down_bytes'].sum()
    distinct_dsts = grouped['dst_ip'].nunique()
    distinct_ports = grouped['port'].nunique()

    mean_up = grouped['up_bytes'].mean()
    std_up = grouped['up_bytes'].std()
    median_up = grouped['up_bytes'].median()

    mean_down = grouped['down_bytes'].mean()
    std_down = grouped['down_bytes'].std()
    median_down = grouped['down_bytes'].median()

    proto_counts = df_internal.groupby('src_ip')['proto'].value_counts().unstack(fill_value=0)
    if 'tcp' not in proto_counts:
        proto_counts['tcp'] = 0
    if 'udp' not in proto_counts:
        proto_counts['udp'] = 0

    per_ip = pd.DataFrame({
        'total_flows': total_flows,
        'total_up_bytes': total_up,
        'total_down_bytes': total_down,
        'distinct_dsts': distinct_dsts,
        'distinct_ports': distinct_ports,
        'mean_up_bytes': mean_up,
        'std_up_bytes': std_up,
        'median_up_bytes': median_up,
        'mean_down_bytes': mean_down,
        'std_down_bytes': std_down,
        'median_down_bytes': median_down,
        'tcp_flows': proto_counts['tcp'],
        'udp_flows': proto_counts['udp'],
    })

    df_sorted = df_internal.sort_values(['src_ip', 'timestamp'])
    intervals = df_sorted.groupby('src_ip')['timestamp'].diff()
    mean_interval = intervals.groupby(df_sorted['src_ip']).mean()
    std_interval = intervals.groupby(df_sorted['src_ip']).std()

    avg_ratio = (total_up / total_down.replace(0, np.nan))

    per_ip['mean_interval'] = mean_interval
    per_ip['std_interval'] = std_interval
    per_ip['up_down_ratio'] = avg_ratio

    for ip in sorted(per_ip.index):
        row = per_ip.loc[ip]
        up = total_up[ip]
        down = total_down[ip]
        ratio_val = up / down if down > 0 else float('inf')

        print(f'\n--- IP: {ip} ---')
        print(f'  Total flows: {int(row["total_flows"])}')
        print(f'  TCP/UDP: {int(row["tcp_flows"])} / {int(row["udp_flows"])}')
        print(f'  Total up_bytes: {up}')
        print(f'  Total down_bytes: {down}')
        print(f'  Up/down ratio: {ratio_val:.2f}')
        print(f'  Distinct dst IPs: {int(row["distinct_dsts"])}')
        print(f'  Distinct ports: {int(row["distinct_ports"])}')
        print(f'  Mean up_bytes per flow: {row["mean_up_bytes"]:.2f} (std={row["std_up_bytes"]:.2f}, median={row["median_up_bytes"]:.2f})')
        print(f'  Mean down_bytes per flow: {row["mean_down_bytes"]:.2f} (std={row["std_down_bytes"]:.2f}, median={row["median_down_bytes"]:.2f})')
        print(f'  Mean inter-flow interval (1/100s): {row["mean_interval"]:.2f} (std={row["std_interval"]:.2f})')

    per_ip_countries = {}
    for ip in sorted(per_ip.index):
        ip_flows = df_internal[df_internal['src_ip'] == ip]
        public_dsts = ip_flows[~ip_flows['dst_ip'].apply(
            lambda x: ipaddress.IPv4Address(x).is_private
        )]['dst_ip']
        if len(public_dsts) == 0:
            continue
        countries = public_dsts.apply(lambda x: get_country(x, reader_country))
        country_counts = countries.value_counts().to_dict()
        per_ip_countries[ip] = country_counts
        print(f'\n  Destination countries for {ip}: {country_counts}')

    print()

    port_stats = df_internal.groupby('port').agg(
        flow_count=('src_ip', 'count'),
        total_up_bytes=('up_bytes', 'sum'),
        total_down_bytes=('down_bytes', 'sum'),
    ).sort_values('flow_count', ascending=False)

    print('Per-port statistics (top 15):')
    for port, row in port_stats.head(15).iterrows():
        print(f'  Port {port}: {int(row["flow_count"])} flows, '
              f'up={int(row["total_up_bytes"])}, down={int(row["total_down_bytes"])}')
    print()

    return per_ip, per_ip_countries


def analyze_dns_https(df_internal):
    print('=' * 60)
    print('4. DNS vs HTTPS')
    print('=' * 60)

    dns = df_internal[df_internal['port'] == 53]
    https = df_internal[df_internal['port'] == 443]

    dns_per_ip = dns.groupby('src_ip').agg(
        dns_flows=('up_bytes', 'count'),
        dns_up_bytes=('up_bytes', 'sum'),
        dns_down_bytes=('down_bytes', 'sum'),
        dns_mean_up=('up_bytes', 'mean'),
    )

    https_per_ip = https.groupby('src_ip').agg(
        https_flows=('up_bytes', 'count'),
        https_up_bytes=('up_bytes', 'sum'),
        https_down_bytes=('down_bytes', 'sum'),
        https_mean_up=('up_bytes', 'mean'),
    )

    per_ip = dns_per_ip.join(https_per_ip, how='outer', on='src_ip').fillna(0)

    per_ip['dns_https_flow_ratio'] = per_ip['dns_flows'] / per_ip['https_flows'].replace(0, np.nan)
    per_ip['dns_https_up_ratio'] = per_ip['dns_up_bytes'] / per_ip['https_up_bytes'].replace(0, np.nan)
    per_ip['https_up_down_ratio'] = per_ip['https_up_bytes'] / per_ip['https_down_bytes'].replace(0, np.nan)

    print('\nPer-IP DNS and HTTPS breakdown:')
    for ip in sorted(per_ip.index):
        row = per_ip.loc[ip]
        flow_ratio = row['dns_https_flow_ratio']
        up_ratio = row['dns_https_up_ratio']
        print(f'  {ip}:')
        print(f'    DNS:  {int(row["dns_flows"]):>6} flows, up={int(row["dns_up_bytes"]):>12}, down={int(row["dns_down_bytes"]):>12}')
        print(f'    HTTPS:{int(row["https_flows"]):>6} flows, up={int(row["https_up_bytes"]):>12}, down={int(row["https_down_bytes"]):>12}')
        print(f'    DNS/HTTPS flow ratio: {flow_ratio if np.isfinite(flow_ratio) else "inf":>12}')
        print(f'    DNS/HTTPS up ratio:   {up_ratio if np.isfinite(up_ratio) else "inf":>12}')

    flow_ratios = per_ip['dns_https_flow_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
    up_ratios = per_ip['dns_https_up_ratio'].replace([np.inf, -np.inf], np.nan).dropna()

    print(f'\nGlobal DNS/HTTPS flow ratio stats (across IPs): '
          f'mean={flow_ratios.mean():.4f}, std={flow_ratios.std():.4f}, '
          f'median={flow_ratios.median():.4f}')
    print(f'Global DNS/HTTPS up ratio stats (across IPs): '
          f'mean={up_ratios.mean():.4f}, std={up_ratios.std():.4f}, '
          f'median={up_ratios.median():.4f}')
    print()

    return per_ip


def analyze_destinations(df_internal, reader_country, reader_asn, private_network):
    print('=' * 60)
    print('5. DESTINATION COUNTRIES & ASNs (EXTERNAL)')
    print('=' * 60)

    external = df_internal[~df_internal['dst_ip'].apply(
        lambda x: ipaddress.IPv4Address(x) in private_network
    )]

    if len(external) == 0:
        print('No external destinations found.')
        print()
        return

    external = external.copy()
    external['dst_country'] = external['dst_ip'].apply(lambda x: get_country(x, reader_country))
    external['dst_asn'] = external['dst_ip'].apply(lambda x: get_asn(x, reader_asn))

    country_counts = external.groupby('dst_country').agg(
        flows=('src_ip', 'count'),
        up_bytes=('up_bytes', 'sum'),
        down_bytes=('down_bytes', 'sum'),
    ).sort_values('flows', ascending=False)

    print(f'\nTop destination countries by flow count:')
    for cc, row in country_counts.iterrows():
        print(f'  {cc}: {int(row["flows"])} flows, '
              f'up={int(row["up_bytes"])}, down={int(row["down_bytes"])}')

    asn_counts = external.groupby('dst_asn').agg(
        flows=('src_ip', 'count'),
        up_bytes=('up_bytes', 'sum'),
        down_bytes=('down_bytes', 'sum'),
    ).sort_values('flows', ascending=False)

    print(f'\nTop destination ASNs by flow count:')
    for asn, row in asn_counts.head(10).iterrows():
        print(f'  ASN {asn}: {int(row["flows"])} flows, '
              f'up={int(row["up_bytes"])}, down={int(row["down_bytes"])}')
    print()


def analyze_external_users(df_external, reader_country, reader_asn):
    print('=' * 60)
    print('6. EXTERNAL USERS TRAFFIC')
    print('=' * 60)

    unique_ips = sorted(df_external['src_ip'].unique())
    print(f'Unique external source IPs: {len(unique_ips)}')
    print(f'External IPs: {unique_ips}')

    print()
    for src_ip in unique_ips:
        ip_data = df_external[df_external['src_ip'] == src_ip].sort_values('timestamp')
        flows = len(ip_data)
        total_up = ip_data['up_bytes'].sum()
        total_down = ip_data['down_bytes'].sum()
        distinct_ports = ip_data['port'].nunique()
        distinct_dsts = ip_data['dst_ip'].nunique()

        intervals = ip_data['timestamp'].diff().dropna()
        mean_int = intervals.mean() if len(intervals) > 0 else 0
        std_int = intervals.std() if len(intervals) > 0 else 0

        ports_used = ip_data.groupby('port').agg(
            flows=('up_bytes', 'count'),
            total_up=('up_bytes', 'sum'),
            total_down=('down_bytes', 'sum'),
        ).sort_values('flows', ascending=False)

        hour_bins = (ip_data['timestamp'] // 360000).value_counts().sort_index()

        print(f'--- {src_ip} ---')
        print(f'  Total flows: {flows}')
        print(f'  Total up_bytes: {total_up}, Total down_bytes: {total_down}')
        up_down = total_up / total_down if total_down > 0 else float('inf')
        print(f'  Up/down ratio: {up_down:.2f}')
        print(f'  Distinct dst IPs: {distinct_dsts}')
        print(f'  Distinct ports: {distinct_ports}')
        print(f'  Mean inter-flow interval (1/100s): {mean_int:.2f} (std={std_int:.2f})')

        print(f'  Port usage:')
        for port, row in ports_used.iterrows():
            print(f'    Port {port}: {int(row["flows"])} flows, '
                  f'up={int(row["total_up"])}, down={int(row["total_down"])}')

        print(f'  Activity per hour (0-23):')
        for h, c in hour_bins.items():
            print(f'    {int(h):>2}: {int(c):>5} flows')
        print()

    ext_stats_rows = []
    for src_ip in sorted(df_external['src_ip'].unique()):
        ip_data = df_external[df_external['src_ip'] == src_ip].sort_values('timestamp')
        flows = len(ip_data)
        intervals = ip_data['timestamp'].diff().dropna()
        ext_stats_rows.append({
            'src_ip': src_ip,
            'flows': flows,
            'total_up_bytes': ip_data['up_bytes'].sum(),
            'total_down_bytes': ip_data['down_bytes'].sum(),
            'distinct_ports': ip_data['port'].nunique(),
            'distinct_dsts': ip_data['dst_ip'].nunique(),
            'mean_interval': intervals.mean() if len(intervals) > 0 else 0,
            'std_interval': intervals.std() if len(intervals) > 0 else 0,
        })
    return pd.DataFrame(ext_stats_rows)


def clean_series(series):
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def print_threshold(desc, s, method='p95'):
    s = clean_series(s)
    if len(s) == 0:
        return
    median = s.median()
    p1, p5, p95, p99 = s.quantile(0.01), s.quantile(0.05), s.quantile(0.95), s.quantile(0.99)
    mn, mx = s.min(), s.max()
    mean_val, std_val = s.mean(), s.std()

    print(f'  {desc}:')
    print(f'    {len(s)} IPs, min={mn:.2f}, p5={p5:.2f}, median={median:.2f}, '
          f'mean={mean_val:.2f}, std={std_val:.2f}, p95={p95:.2f}, p99={p99:.2f}, max={mx:.2f}')

    if method == 'p95':
        print(f'    >> threshold = 95th percentile = {p95:.2f}')
    elif method == 'p99':
        print(f'    >> threshold = 99th percentile = {p99:.2f}')
    elif method == 'p5':
        print(f'    >> threshold = 5th percentile (flag below this) = {p5:.2f}')
    elif method == 'mean2std':
        t = mean_val + 2 * std_val
        print(f'    >> threshold = mean + 2σ = {t:.2f}')
    elif method == 'mean3std':
        t = mean_val + 3 * std_val
        print(f'    >> threshold = mean + 3σ = {t:.2f}')


def print_global_summaries(per_ip_stats, dns_https, per_ip_countries, df_external, private_network):
    print('=' * 60)
    print('7. GLOBAL SUMMARY — RULE THRESHOLDS')
    print('   Hybrid approach:')
    print('     - Global absolute thresholds (mean+kσ from training) for ratios & packet sizes')
    print('     - Per-IP fixed 3x multiplier for volume metrics (flows, bytes, destinations)')
    print('     - Categorical baseline for anomalous destinations')
    print('=' * 60)

    print('\n--- Rule: Internal BotNet ---')
    print('  Approach: Per-IP deviation with fixed 3x multiplier')
    print('  Signal: test_flows > 3x train_flows AND > 500')
    print('  Signal: test_dsts   > 3x train_dsts   AND > 20')
    print_threshold('Training flow count reference', per_ip_stats['total_flows'], 'p95')
    print_threshold('Training unique dst IPs reference', per_ip_stats['distinct_dsts'], 'p95')

    print('\n--- Rule: Data Exfiltration via DNS ---')
    print('  Approach: Global threshold for DNS/HTTPS ratio + per-IP 3x for DNS volume')
    print(f'  Global threshold: dns_https_flow_ratio > {0.20} (above training max of 0.19)')
    print('  Per-IP: test_dns_flows > 3x train_dns_flows AND > 100')
    print('  Per-IP: test_dns_up > 3x train_dns_up AND > 10,000')
    print_threshold('Training DNS/HTTPS ratio reference', dns_https['dns_https_flow_ratio'], 'mean2std')

    print('\n--- Rule: Data Exfiltration via HTTPS ---')
    print('  Approach: Global threshold for up/down ratio + per-IP 3x for HTTPS volume')
    print(f'  Global threshold: https_up_down_ratio > {0.12} (training mean + 3σ, mean≈0.11, σ≈0.003)')
    print('  Per-IP: test_https_up > 3x train_https_up AND > 100,000')
    print('  Ratio signal requires AND with volume surge')

    print('\n--- Rule: C&C via DNS ---')
    print('  Approach: Global threshold for DNS packet size + per-IP 3x for DNS flows')
    print(f'  Global threshold: dns_mean_up > {217} bytes (training mean + 5σ, mean≈199.9, σ≈3.47)')
    print('  Per-IP: test_dns_flows > 3x train_dns_flows AND > 100')
    print_threshold('Training DNS mean upload reference', dns_https['dns_mean_up'], 'mean3std')

    print('\n--- Rule: Anomalous External Destinations ---')
    print('  Approach: Global categorical baseline')
    print('  Baseline: countries seen by any IP in training.')
    print('  Rule: flag IP if test data contacts a country NOT in training, with >= 3 flows.')
    all_countries = set()
    for countries in per_ip_countries.values():
        all_countries.update(countries.keys())
    print(f'  {len(per_ip_countries)} IPs have country data from training.')
    print(f'  Global country set seen in training ({len(all_countries)}): {sorted(all_countries)}')

    print('\n--- Rule: External User Behavior ---')
    print('  Approach: Per-IP temporal (off-hours vs training max) + global CV floor')
    ext_stats = []
    for src_ip in sorted(df_external['src_ip'].unique()):
        ip_data = df_external[df_external['src_ip'] == src_ip].sort_values('timestamp')
        flows = len(ip_data)
        intervals = ip_data['timestamp'].diff().dropna()
        hours = (ip_data['timestamp'] // 360000).unique()
        mean_int = intervals.mean() if len(intervals) > 0 else 0
        std_int = intervals.std() if len(intervals) > 0 else 0
        cv = std_int / mean_int if mean_int > 0 else np.nan
        ext_stats.append({
            'src_ip': src_ip,
            'flows': flows,
            'cv': cv,
            'hours': sorted(hours),
        })
    ext_df = pd.DataFrame(ext_stats)
    cvs = ext_df['cv'].dropna()
    min_cv = cvs.min()
    print(f'  Global CV floor (minimum across all training IPs): {min_cv:.2f}')
    print(f'  Rule: flag external IP if test_CV < {min_cv:.2f} (more regular than any training IP)')
    print(f'        OR off-hours > 50% of test flows outside training active hours')
    print(f'        OR both (AND logic)')



def generate_graphs(dns_https, df_internal, reader_country, private_network):
    print('=' * 60)
    print('SAVING GRAPHS')
    print('=' * 60)

    flow_ratios = dns_https['dns_https_flow_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(8, 5))
    plt.hist(flow_ratios, bins=30, edgecolor='black')
    plt.xlabel('DNS/HTTPS flow count ratio')
    plt.ylabel('Frequency (number of IPs)')
    plt.title('Distribution of DNS/HTTPS flow ratio per IP')
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'dns_https_flow_ratio_hist_train.png')
    plt.close()
    print('  Saved graphs/dns_https_flow_ratio_hist_train.png')

    external = df_internal[~df_internal['dst_ip'].apply(
        lambda x: ipaddress.IPv4Address(x) in private_network
    )]
    if len(external) > 0:
        external = external.copy()
        external['dst_country'] = external['dst_ip'].apply(lambda x: get_country(x, reader_country))
        top = external['dst_country'].value_counts().head(15)

        plt.figure(figsize=(12, 5))
        top.plot(kind='bar')
        plt.xlabel('Country')
        plt.ylabel('Flow count')
        plt.title('Top 15 destination countries')
        plt.tight_layout()
        plt.savefig(GRAPHS_DIR / 'top_countries_train.png')
        plt.close()
        print('  Saved graphs/top_countries_train.png')

    intervals = []
    for src_ip in df_internal['src_ip'].unique():
        ip_data = df_internal[df_internal['src_ip'] == src_ip].sort_values('timestamp')
        diffs = ip_data['timestamp'].diff().dropna()
        if len(diffs) > 0:
            intervals.append(diffs.mean())

    plt.figure(figsize=(8, 5))
    plt.hist(intervals, bins=30, edgecolor='black')
    plt.xlabel('Mean inter-flow interval (1/100s)')
    plt.ylabel('Frequency (number of IPs)')
    plt.title('Distribution of mean inter-flow intervals per IP')
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'interval_hist_train.png')
    plt.close()
    print('  Saved graphs/interval_hist_train.png')

    port_stats = dns_https[['dns_flows', 'https_flows']]
    plt.figure(figsize=(8, 6))
    plt.scatter(port_stats['https_flows'], port_stats['dns_flows'], alpha=0.7)
    plt.xlabel('HTTPS flow count')
    plt.ylabel('DNS flow count')
    plt.title('DNS vs HTTPS flows per IP')
    for ip in port_stats.index:
        plt.annotate(str(ip).split('.')[-1], (port_stats.loc[ip, 'https_flows'], port_stats.loc[ip, 'dns_flows']),
                     fontsize=8, alpha=0.7)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / 'dns_vs_https_scatter_train.png')
    plt.close()
    print('  Saved graphs/dns_vs_https_scatter_train.png')

    print()


def main():
    print('Loading data...')
    df_internal, df_external = load_data()
    print(f'  internal_train7: {len(df_internal)} flows')
    print(f'  external_train7: {len(df_external)} flows')
    print()

    print('Loading GeoIP databases...')
    reader_country, reader_asn = load_geoip()
    print('GeoIP databases loaded.')
    print()

    private_network = identify_private_network(df_internal)
    identify_internal_servers(df_internal, private_network)
    per_ip_stats, per_ip_countries = analyze_src_ip_stats(df_internal, reader_country, reader_asn)
    dns_https = analyze_dns_https(df_internal)
    analyze_destinations(df_internal, reader_country, reader_asn, private_network)
    ext_stats_df = analyze_external_users(df_external, reader_country, reader_asn)
    generate_graphs(dns_https, df_internal, reader_country, private_network)
    print_global_summaries(per_ip_stats, dns_https, per_ip_countries, df_external, private_network)

    reader_country.close()
    reader_asn.close()
    print('Done.')


if __name__ == '__main__':
    main()
