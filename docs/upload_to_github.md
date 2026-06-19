# Загрузка проекта на GitHub

## 1. Создать репозиторий

1. Откройте GitHub.
2. Нажмите `New repository`.
3. Назовите репозиторий, например `credit-scoring`.
4. Не добавляйте README на сайте, он уже есть в папке проекта.

## 2. Загрузить через терминал

Перейдите в подготовленную папку:

```bash
cd path/to/github_ready_credit_scoring
```

И выполните:

```bash
git init
git add .
git commit -m "Initial credit scoring solution"
git branch -M main
git remote add origin https://github.com/<USERNAME>/<REPOSITORY>.git
git push -u origin main
```

Замените:

- `<USERNAME>` на ваш GitHub-логин;
- `<REPOSITORY>` на имя созданного репозитория.

## 3. Что не загружать

Не добавляйте вручную в GitHub:

- исходные parquet-файлы из `data/`;
- промежуточные файлы из `work/`;
- временные файлы Python;
- дополнительные большие сабмиты.

Эти файлы закрыты через `.gitignore`.

## 4. Проверка перед push

```bash
git status
git diff --cached --stat
```

В коммите должны быть код, README, requirements, инструкция и готовый `submissions/submission.csv`.
