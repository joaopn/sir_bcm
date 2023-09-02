import sys
import socket
from dask.distributed import LocalCluster, Client
import time

def spawn_dask_cluster(n_workers):
    # Get the machine's IP address
    ip_address = socket.gethostbyname(socket.gethostname())

    # Start a local cluster binding to all network interfaces
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
        host="0.0.0.0",  # This is the change, to bind to all network interfaces
        scheduler_port=8786,  # Explicitly set the scheduler port
        dashboard_address=f':8787'
    )

    # Connect a client to the cluster
    client = Client(cluster)

    # Print the scheduler and dashboard addresses using the machine's IP
    print(f"Scheduler address: tcp://{ip_address}:8786")
    print(f"Dashboard address: http://{ip_address}:8787/status")

    # Keep the cluster running (for demonstration purposes)
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nShutting down Dask cluster...")
        client.close()
        cluster.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify the number of workers/cores as an argument.")
        sys.exit(1)

    n_workers = int(sys.argv[1])
    spawn_dask_cluster(n_workers)
