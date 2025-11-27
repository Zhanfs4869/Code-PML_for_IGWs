rm -f KEz.dat
for i in {0..60}; do
    python3 Trial_PML_10.py $i
    echo "================ Finished i=$i ================"
done