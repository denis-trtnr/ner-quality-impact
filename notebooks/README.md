If you want to use notebook with cluster ressources:

1. Start interactive bash session
    ```bash
    srun -K \
      --job-name jn \
      -p RTXA6000,A100-40GB,A100-80GB,H100,H100-SLT,RTX3090,batch,H200,A100-SDS,A100-PCI,L40S \
      --gpus=1 \
      --mem=48G \
      --container-mounts=/netscratch:/netscratch,/home/$USER:/home/$USER \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.07-py3.sqsh \
      --container-workdir=$(pwd) \
      --time 02:00:00 \
      --immediate=1000 \
      --pty bash
    ```
2. Start Jupyter Notebook Server
    ```bash
    echo "Jupyter starting at ... http://${HOSTNAME}.kl.dfki.de:8880" && jupyter notebook --ip=0.0.0.0 --port=8880 \
        --allow-root --no-browser --config /home/dtrautner/.jupyter/jupyter_notebook_config.json \
        --notebook-dir /home/dtrautner/dev/ner-quality-impact
    ```
3. Copy the url from the terminal in the following format
    ```bash
    http://${HOSTNAME}.kl.dfki.de:8880/?token=<token>
    ```
4. Add Server as Kernal in VS Code