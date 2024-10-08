hdir=$(pwd)
cd ..

mkdir datasets

cd datasets

kaggle competitions download -c lmsys-chatbot-arena
unzip lmsys-chatbot-arena.zip -d lmsys-chatbot-arena
rm lmsys-chatbot-arena.zip

kaggle datasets download -d conjuring92/lmsys-mix-v110
unzip lmsys-mix-v01.zip -d ./lmsys-mix-v110
rm lmsys-mix-v110.zip

kaggle datasets download -d conjuring92/lmsys-mix-v205
unzip lmsys-mix-v01.zip -d ./lmsys-mix-v110
rm lmsys-mix-v110.zip

cd $hdir