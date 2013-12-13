set terminal png
set output 'roc.png'
set size square

set title "ROC"
set key bottom right
set xlabel 'False-positive Rate'
set ylabel 'True-positive Rate'

plot "predmond-rocdata-bernoulli-fold0.txt" using 1:2 with lines title "bernoulli",\
     "predmond-rocdata-gaussian-fold0.txt"  using 1:2 with lines title "gaussian" ,\
     "predmond-rocdata-histogram-fold0.txt" using 1:2 with lines title "histogram"
