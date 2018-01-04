# Data Science

A markdown website with everything I know about data science. Run with hugo locally: https://gohugo.io/. From the command line with: 'hugo serve'.

## Running locally
To serve this website locally you will need to (1.) have Hugo and a Sass-compiler installed and (2.) compile the Sass and serve the hugo site.

### Linux / Mac
Use your favourite package manager to install [`fswatch`](https://github.com/emcrisostomo/fswatch), [`sassc`](https://github.com/sass/sassc) and [hugo](https://gohugo.io). (All of these are available on Homebrew.)

In the root directory of the project, run `bash serve.sh`.

### Windows
First, install [hugo](https://gohugo.io) and [node.js](https://nodejs.org) through a package manager or manually. Next, install `node-sass` globally from the command line:

```
npm install -g node-sass
```

Restart your terminal application. Make sure that `node-sass --version` works without errors. You will unfortunately have to recompile the css every time you change the scss file, using the following command:

```
node-sass scss/main.scss > static/css/style.css
```

Open a new terminal instance and start the hugo server:

```
hugo serve
```

Cheers. A.
