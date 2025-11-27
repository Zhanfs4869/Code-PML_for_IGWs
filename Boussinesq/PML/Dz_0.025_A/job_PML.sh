rm -f KEz.dat
for i in {0..1000}; do
    python3 Trial_PML.py $i
    echo "================ Finished i=$i ================"
done