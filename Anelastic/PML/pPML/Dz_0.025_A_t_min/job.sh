rm -f dt_min.dat
for i in {0..100}; do
    python3 test.py $i
    echo "================ Finished i=$i ================"
done