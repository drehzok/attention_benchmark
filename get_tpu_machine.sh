#!/bin/bash

# --- CONFIGURATION ---
export PROJECT_ID="tpu-attention-benchmark"
# We try the "4-chip" variants to maximize success chances
# Format: "ZONE | ACCELERATOR_TYPE | VERSION"

CANDIDATES=(
    # --- v5e (Lite) Candidates (Most likely to succeed) ---
    "us-central1-a|v5litepod-4|tpu-ubuntu2204-base"
    "us-west1-c|v5litepod-4|tpu-ubuntu2204-base"
    "us-west4-a|v5litepod-4|tpu-ubuntu2204-base"
    "us-east1-c|v5litepod-4|tpu-ubuntu2204-base"
    "europe-west4-b|v5litepod-4|tpu-ubuntu2204-base"  # You had quota issues with 8, 4 might work
    "asia-southeast1-c|v5litepod-4|tpu-ubuntu2204-base"

    # --- v6e (Trillium) Candidates ---
    "europe-west4-a|v6e-4|v2-alpha-tpuv6e"
    "us-east1-d|v6e-4|v2-alpha-tpuv6e"
    "us-east5-b|v6e-4|v2-alpha-tpuv6e"

    # --- v4 Candidates (Hardest to get on Spot) ---
    "us-central2-b|v4-8|tpu-ubuntu2204-base"
)

# --- THE LOOP ---
for candidate in "${CANDIDATES[@]}"; do
    IFS="|" read -r ZONE TYPE VERSION <<< "$candidate"
    
    TPU_NAME="scanner-${TYPE}-${ZONE}"
    
    echo "----------------------------------------------------------------"
    echo "Trying: $TYPE in $ZONE..."
    echo "----------------------------------------------------------------"

    # 1. Check if we already have one running here (safety check)
    EXISTS=$(gcloud compute tpus tpu-vm list --zone=$ZONE --project=$PROJECT_ID --format="value(name)" --filter="name:$TPU_NAME")
    
    if [ -n "$EXISTS" ]; then
        echo "Found existing instance $TPU_NAME. Checking status..."
        STATUS=$(gcloud compute tpus tpu-vm list --zone=$ZONE --project=$PROJECT_ID --format="value(state)")
        if [ "$STATUS" == "READY" ]; then
            echo "✅ SUCCESS! Machine is READY."
            echo "SSH Command: gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT_ID"
            exit 0
        else
            echo "Instance exists but is $STATUS. Deleting to retry..."
            gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --project=$PROJECT_ID --quiet
        fi
    fi

    # 2. Attempt Creation
    gcloud compute tpus tpu-vm create $TPU_NAME \
        --zone=$ZONE \
        --accelerator-type=$TYPE \
        --version=$VERSION \
        --project=$PROJECT_ID \
        --spot \
        --quiet > /dev/null 2>&1

    # 3. Check Result
    if [ $? -eq 0 ]; then
        echo "✅ CREATION ACCEPTED! Waiting for READY status..."
        
        # Wait loop (max 60 seconds)
        for i in {1..12}; do
            STATUS=$(gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$ZONE --project=$PROJECT_ID --format="value(state)" 2>/dev/null)
            
            if [ "$STATUS" == "READY" ]; then
                echo ""
                echo "🎉 WINNER FOUND!"
                echo "Zone: $ZONE"
                echo "Type: $TYPE"
                echo "Name: $TPU_NAME"
                echo "----------------------------------------------------------------"
                echo "Run this to connect:"
                echo "gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT_ID"
                exit 0
            elif [ "$STATUS" == "PREEMPTED" ] || [ "$STATUS" == "STOPPED" ]; then
                echo "❌ FAILED (Preempted immediately)."
                break
            fi
            echo -n "."
            sleep 5
        done
        
        # If we get here, it failed or took too long
        echo "❌ Deleting failed instance..."
        gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --project=$PROJECT_ID --quiet &
    else
        echo "❌ FAILED (Capacity/Quota)."
    fi
done

echo "😭 All regions exhausted. No Spot capacity available right now."
