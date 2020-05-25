source $(pipenv --venv)/bin/activate

video="/home/renan/Datasets/D3PERJ/unsorted/14/14.mp4"
ref="/home/renan/Datasets/D3PERJ/unsorted/14/14.rttm"

outdir="/home/renan/SpeechActionClassifier/export/14-model-strat-step"
refsfile="$outdir/14-refs.scp"
refsdir="$outdir/refs"
sysfile="$outdir/14-sys.scp"
sysdir="$outdir/sys"

mkdir "$refsdir" -p
mkdir "$sysdir" -p

rm $sysfile $refsfile -f

# Model - Best Loss
for model in "loss" "accuracy"; do
    for strat in "mean" "median" "freq" "gaussw" "gaussf"; do
        for step in 1 3 5 15; do
            outname="14-$model-$strat-$step.rttm"

            # Copy reference rttm
            cp "$ref" "$refsdir/$outname"
            sed -i 's/ 14 / '"14-$model-$strat-$step"' /g' "$refsdir/$outname"
            echo "$refsdir/$outname" >> $refsfile

            # Diarize
            python ./diarize.py "$video" "/home/renan/SpeechActionClassifier/export/2020-05-18-1528/model.best-$model.h5" -o "$sysdir/$outname" --diar --conf --strat "$strat" --step "$step"

            # Replace video name in generated file to match reference
            sed -i 's/ 14 / '"14-$model-$strat-$step"' /g' "$sysdir/$outname"
            echo "$sysdir/$outname" >> $sysfile
        done
    done
done

# Display metrics
python ./tools/dscore/score.py --step 0.33 --collar 0.1 -R "$refsfile" -S "$sysfile"
