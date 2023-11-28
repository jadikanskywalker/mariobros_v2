git config --global user.email "awesomejaden.hicks@gmail.com"
git config --global user.name "jadikanskywalker"

git add -f .
git commit -m "New commit"

git remote remove origin
git remote add origin https://jadikanskywalker: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@github.com/jadikanskywalker/mariobros_v2.git

git push -u origin main
