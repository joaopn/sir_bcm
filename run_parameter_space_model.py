import argparse
import sir_bcm as sbcm
import pandas as pd
import time

def run_analysis(N, max_timesteps, epsilons, mus, n_trials, stop_tol, cluster_ip):
    df_results = sbcm.dask.parallel_analysis(N, max_timesteps, epsilons, mus, n_trials, stop_tol=stop_tol, cluster_ip=cluster_ip)
    return df_results

def main():
    parser = argparse.ArgumentParser(description="Execute analysis with SIR_BCM and save results to CSV.")

    parser.add_argument("--filename", required=True, help="Prefix for the output CSV file.")
    parser.add_argument("--N", type=int, nargs='+', required=True, help="List of values for N.")
    parser.add_argument("--max_timesteps", type=int, default=1000, help="Value for max_timesteps. Default is 1000.")
    parser.add_argument("--n_trials", type=int, default=2, help="Value for n_trials. Default is 2.")
    parser.add_argument("--stop_tol", type=float, default=1e-6, help="Value for stop_tol. Default is 1e-6.")
    parser.add_argument("--epsilons", type=float, nargs='+', default=[0.1, 0.3], help="List of values for epsilons. Default is [0.1, 0.3].")
    parser.add_argument("--mus", type=float, nargs='+', default=[0.05, 0.3], help="List of values for mus. Default is [0.05, 0.3].")
    parser.add_argument("--cluster_ip", required=True, help="IP address of the Dask cluster.")

    args = parser.parse_args()

    for n in args.N:
        start_time = time.time()
        
        df_results = run_analysis(n, args.max_timesteps, args.epsilons, args.mus, args.n_trials, args.stop_tol, args.cluster_ip)
        
        elapsed_time = time.time() - start_time
        print(f"Time for N={n}: {elapsed_time:.2f} seconds")
        
        # Format CSV filename
        csv_filename = f"{args.filename}_N{n}_maxtimesteps{args.max_timesteps}_ntrials{args.n_trials}_stoptol{args.stop_tol}.csv"
        
        # Save results to CSV
        df_results.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    main()
