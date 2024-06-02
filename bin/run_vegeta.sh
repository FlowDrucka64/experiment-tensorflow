kubectl run vegeta --rm --attach --restart=Never -n faasm --image="peterevans/vegeta" -- sh -c \
"echo 'POST https://www.example.com' | vegeta attack -rate=10 -duration=30s | tee results.bin | vegeta report"
