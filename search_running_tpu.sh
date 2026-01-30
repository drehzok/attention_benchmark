# 1. Restore Project ID
export PROJECT_ID="tpu-attention-benchmark"

# 2. Define the zones we were testing
ZONES=(
  "us-central2-b"    # v4
  "us-central1-a"    # v5e
  "us-west1-c"       # v5e
  "us-west4-a"       # v5e
  "us-east1-c"       # v5e
  "europe-west4-b"   # v5e (The most likely winner)
  "asia-southeast1-c" # v5e
  "europe-west4-a"   # v6e
  "us-east1-d"       # v6e
  "us-east5-b"       # v6e
)

echo "=========================================="
echo "      SCANNING FOR SUCCESSFUL TPUS        "
echo "=========================================="

# 3. Loop through every zone
for z in "${ZONES[@]}"; do
  # We use '|| true' so the script doesn't stop if a zone is unreachable
  RESULT=$(gcloud compute tpus tpu-vm list --zone=$z --project=$PROJECT_ID 2>/dev/null)
  
  if [ -n "$RESULT" ]; then
    echo "[$z]: FOUND SOMETHING!"
    echo "$RESULT"
    echo "------------------------------------------"
  else
    echo "[$z]: Empty."
  fi
done

echo ""
echo "=========================================="
echo "      CHECKING ORPHANED DISKS             "
echo "=========================================="
# 4. Check for disks (Disks are global-view, so one command checks all)
gcloud compute disks list --project=$PROJECT_ID
