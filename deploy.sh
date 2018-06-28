rm -rf tc
cp models.py tc
cp main.py tc
cp utils.py tc
cp data/data.txt tc
cp config tc
cp tmp_test_content tc
cp tmp_test_content.txt tc
tar -zcvf tc.tar.gz tc
ssh neukg6@219.216.64.90 "cd /home/neukg6/anaconda3/a; rm -rf tc; mkdir tc; exit"
scp tc.tar.gz neukg6@219.216.64.90:/home/neukg6/anaconda3/a/tc/
ssh neukg6@219.216.64.90 "cd /home/neukg6/anaconda3/a/tc/; tar -zxvf tc.tar.gz; rm -rf tc.tar.gz; exit;"
rm -rf tc.tar.gz
