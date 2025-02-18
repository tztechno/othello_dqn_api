pip install fastapi uvicorn torch numpy requests

uvicorn api:app --reload

http://localhost:8000/docs


echo "# othello_dqn_api" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/tztechno/othello_dqn_api.git
git add
git push -f origin main