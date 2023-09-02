import argparse
import sir_bcm as sbcm
import pandas as pd

def run_analysis(N, max_timesteps, epsilons, mus, n_trials, n_cores, stop_tol):
    df_results = sbcm.dask.parallel_analysis(N, max_timesteps, epsilons, mus, n_trials, n_cores=n_cores, stop_tol=stop_tol)
    return df_results

def main():
    parser = argparse.ArgumentParser(description="Execute analysis with SIR_BCM and save results to CSV.")

    parser.add_argument("--filename", required=True, help="Prefix for the output CSV file.")
    parser.add_argument("--N", type=int, default=100, help="Value for N. Default is 100.")
    parser.add_argument("--max_timesteps", type=int, default=1000, help="Value for max_timesteps. Default is 1000.")
    parser.add_argument("--n_trials", type=int, default=2, help="Value for n_trials. Default is 2.")
    parser.add_argument("--stop_tol", type=float, default=1e-6, help="Value for stop_tol. Default is 1e-6.")
    parser.add_argument("--epsilons", type=float, nargs='+', default=[0.1, 0.3], help="List of values for epsilons. Default is [0.1, 0.3].")
    parser.add_argument("--mus", type=float, nargs='+', default=[0.05, 0.3], help="List of values for mus. Default is [0.05, 0.3].")
    parser.add_argument("--n_cores", type=int, default=6, help="Value for n_cores. Default is 6.")

    args = parser.parse_args()

    df_results = run_analysis(args.N, args.max_timesteps, args.epsilons, args.mus, args.n_trials, args.n_cores, args.stop_tol)
    
    # Format CSV filename
    csv_filename = f"{args.filename}_N{args.N}_timesteps{args.max_timesteps}_ntrials{args.n_trials}_stoptol{args.stop_tol}.csv"
    
    # Save results to CSV
    df_results.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    main()
