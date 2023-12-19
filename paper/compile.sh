
# Put everythong in a compile directory
mkdir compile
cp -r * compile/.
cd compile

# Add Latexmk file as excecutable
chmod latexmk 744
EXPORT PATH=${PATH}:${BASE}/paper2/compile/latexmk

#Expand apacite
tex apacite.ins

latexmk -f -pdf > compile.log

cd ..

cp compile/main.pdf output/Probing-Intuitive-Physics-Understanding.pdf
cp compile/compile.log output/.

rm -rf compile