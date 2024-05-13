PYFILES=$(git diff --name-only --diff-filter=ACMRT $commithash HEAD | grep .py | xargs)

for file in $PYFILES; do 
    echo $file
    addlicense -s -c 'NVIDIA CORPORATION & AFFILIATES' $file
done