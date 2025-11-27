rm -f KEz.dat
for i in {0..1000}; do
    python3 PML_Anelastic.py $i
    echo "================ Finished i=$i ================"
done