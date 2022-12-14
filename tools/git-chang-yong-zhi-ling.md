# Git常用指令

{% embed url="https://liaoxuefeng.gitee.io/resource.liaoxuefeng.com/git/git-cheat-sheet.pdf" %}

## git 安装

`sudo apt-get install git`

## git 设置

设置用户名 `git config --global user.name "Your Name"`&#x20;

设置邮箱 `git config --global user.email "Your Email"`&#x20;

查看用户名 `git config user.name`&#x20;

查看邮箱 `git config user.email`&#x20;

查看密码 `git config user.password`&#x20;

查看配置信息 `git config --list`&#x20;

设置显示颜色 `git config --global color.ui true`

## 版本库管理

初始化 `git init`&#x20;

添加文件到版本库 `git add <file name>`&#x20;

提交文件到版本库 `git commit -m "some message"`&#x20;

查看版本库当前状态 `git status`&#x20;

查看修改内容 `git diff`&#x20;

查看版本库的历史记录 `git log [--pretty=oneline]`&#x20;

当前版本 `HEAD` 上一个版本 `HEAD^`&#x20;

往前第100个版本 `HEAD~100`&#x20;

回到上个版本 `git reset --hard HEAD^`&#x20;

回到某个版本 `git reset --hard <commit id>`&#x20;

查看命令记录及commit id `git reflog`&#x20;

查看工作区和版本库里最新版本的区别 `git diff HEAD -- <file name>`&#x20;

丢弃工作区的修改，撤回到最近一个commit或add时的状态 `git checkout -- <file name>`&#x20;

撤销暂存区的修改，将其重新放回工作区 `git reset HEAD <file name>`&#x20;

从版本库中删除文件 `git rm <file name>`

## 远程库管理

将本地的版本库与远程库关联(远程库名字为`origin`) `git remote add origin git@github.com:<host name>/<repo name>.git`&#x20;

将本地库的`master`分支上的内容推送到远程库(`-u`会将本地的`master`分支和远程库的`master`分支关联起来，第一次推送时使用) `git push [-u] origin master`&#x20;

查看远程库信息 `git remote [-v]`&#x20;

删除远程库(删除的是本地库与远程库的绑定关系) `git remote rm <remote repo name>`&#x20;

克隆远程库到本地库(`ssh`协议最快) `git clone git@github.com:<host name>/<repo name>.git`

## 分支管理

git中主分支为`master`，`HEAD`指向当前分支，`master`指向提交，当只有一个主分支时，`HEAD`指向`master`：&#x20;

![](<../.gitbook/assets/image (299).png>)

创建分支 `git branch <branch name>`&#x20;

切换到分支 `git checkout <branch name>`或`git switch <branch name>`&#x20;

创建并切换到分支 `git checkout -b <branch name>`或`git switch -c <branch name>`&#x20;

查看分支(当前分支前会有一个`*`号) `git branch`&#x20;

![](<../.gitbook/assets/image (193).png>)

在`dev`分支上进行修改并提交&#x20;

![](<../.gitbook/assets/image (179).png>)

切换回`master`分支 `git checkout master`或`git switch master`&#x20;

![](<../.gitbook/assets/image (995).png>)

将指定分支上的改动合并到`master`分支上(Fast forward模式，删除分支后，会丢掉分支信息) `git merge <branch name>`&#x20;

![](<../.gitbook/assets/image (5).png>)

如果不使用Fast forward模式合并分支，会在merge的时候生成一个commit信息 `git merge --no-ff -m "some message" <branch name>`&#x20;

![](<../.gitbook/assets/image (563).png>)

删除指定分支 `git branch -d <branch name>`&#x20;

强行删除一个没有被合并过的分支 `git branch -D <branch name>`&#x20;

![](<../.gitbook/assets/image (339).png>)

如果在两个分支上分别有新的提交，git无法快速合并，需要解决冲突&#x20;

![](<../.gitbook/assets/image (180).png>)

手动编辑解决冲突后，重新提交&#x20;

![](<../.gitbook/assets/image (53).png>)

查看分支的合并情况 `git log --graph [--pretty=oneline] [--abbrev-commit]`&#x20;

把本地未push的分叉提交历史整理成直线`git rebase`

### Bug分支

保存工作区的现场 `git stash`&#x20;

查看保存的工作区现场 `git stash list`&#x20;

恢复工作区现场 `git stash apply`&#x20;

删除保存的工作区现场 `git stash drop`&#x20;

恢复并删除工作区现场 `git stash pop`&#x20;

恢复指定的stash `git stash apply stash@{<stash index>}`&#x20;

复制指定的提交到当前分支上 `git cherry-pick <commit id>`

### 多人协作相关

一般而言，`master`分支是主分支，需要与远程库同步；`dev`分支是开发分支，团队成员需要在该分支工作，应与远程库同步；Bug分支用于在本地修复bug，无需推到远程库；feature分支是否推到远程库取决于团队是否在其上合作开发。

### 抓取分支

`git clone`后，一般只能看到本地的master分支&#x20;

创建远程`origin`的分支到本地 `git checkout -b <branch name> origin/<branch name>`&#x20;

推送自己的修改 `git push origin <branch name>`&#x20;

如果推送失败，是因为远程分支比本地更新，用`git pull`将最新的提交抓取下来(如果没有指定本地分支与远程分支的链接，`git pull`会失败，需要`git branch --set-upstream-to=origin/<branch name> <branch name>`)&#x20;

解决冲突后，就可以`gt push origin <branch name>`推送了

## 标签管理

给最新提交的commit打标签 `git tag <tag>`&#x20;

给某次提交的commit打标签 `git tag <tag> <commit id>`&#x20;

查看标签 `git tag`&#x20;

查看标签信息 `git show <tag>`&#x20;

创建带有说明的标签 `git tag -a <tag> -m "some message" <commit id>`&#x20;

删除标签 `git tag -d <tag>`&#x20;

推送某个标签到远程库 `git push origin <tag>`&#x20;

推送所有标签到远程库 `git push origin --tags`&#x20;

删除远程标签 `git push origin :refs/tags/<tag>`
