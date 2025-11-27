rm -f KEz.dat
for i in {0..119}; do
    python3 Trial_Ra.py $i
    echo "================ Finished i=$i ================"
done