tar -zcvf tc.tar.gz --exclude __pycache__ --exclude test.py --exclude cut.py --exclude data/ --exclude scripts --exclude deploy.sh --exclude "data/adult_content.txt" --exclude "data/normal_content.txt" --exclude "*.pyc" --exclude "checkpoint*" --exclude "*.tar.gz" --exclude "*.git*" .
ssh neukg6@219.216.64.90 "cd /home/neukg6/anaconda3/a; rm -rf tc; mkdir tc; exit"
scp tc.tar.gz neukg6@219.216.64.90:/home/neukg6/anaconda3/a/tc/
ssh neukg6@219.216.64.90 "cd /home/neukg6/anaconda3/a/tc/; tar -zxvf tc.tar.gz; rm -rf tc.tar.gz; exit;"
rm -rf tc.tar.gz
