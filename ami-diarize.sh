echo "Diarizando closeups do diretório "$1

audiodir="$1/audio"
export videodir="$1/video"
export outdir="./export/diarized/"

files=($(cd $videodir && find . -regextype posix-extended -regex '.*Closeup[1-4]{1}\.avi' -printf "%P\n"));
audio=($(cd $audiodir && find . -iname '*.wav' -printf "%P\n"))

source  $(pipenv --venv)/bin/activate
printf '%s\n' "${files[@]}" | xargs -n 1 -P 4 -d $'\n' sh -c 'for arg do python3 ./test.py "$videodir/$arg"; python3 ./visualize.py "$videodir/$arg" "$videodir/$arg.diar" "$outdir/${arg%.*}.diarized.mp4"; done' _

ffmpeg -i "$outdir/${files[0]%.*}.diarized.mp4" -i "$outdir/${files[1]%.*}.diarized.mp4" -i "$outdir/${files[2]%.*}.diarized.mp4" -i "$outdir/${files[3]%.*}.diarized.mp4" -i "$audiodir/$audio"\
       	-filter_complex "\
		nullsrc=size=704x576 [base];\
		[0:v] setpts=PTS-STARTPTS, scale=352x288 [upperleft];\
		[1:v] setpts=PTS-STARTPTS, scale=352x288 [upperright];\
		[2:v] setpts=PTS-STARTPTS, scale=352x288 [lowerleft];\
		[3:v] setpts=PTS-STARTPTS, scale=352x288 [lowerright];\
		[base][upperleft] overlay=shortest=1 [tmp1];\
		[tmp1][upperright] overlay=shortest=1:x=352 [tmp2];\
		[tmp2][lowerleft] overlay=shortest=1:y=288 [tmp3];\
		[tmp3][lowerright] overlay=shortest=1:x=352:y=288\
		" -shortest -strict -2 "$outdir/$(basename $1).mp4"
