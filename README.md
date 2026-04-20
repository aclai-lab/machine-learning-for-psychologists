# machine-learning-for-psychologists
This repository contains all the material for the machine learning course taught at the Facultad de Medicina de la Universidad Católica de Murcia

During these sessions, you will learn how to write a complete machine learning pipeline for analysing the typical psychologists data leveraging the [Sole.jl](https://github.com/aclai-lab/Sole.jl) framework.

## Setup

### Julia programming language

If you are using Windows, you can download the installer from [the official website](https://julialang.org/downloads/manual-downloads).
During the installation procedure, we suggest clicking the box "Add Julia to PATH".

If you are using macOS or Linux, you can install Julia by running the following in your terminal:
```
curl -fsSL https://install.julialang.org | sh
```
This will install the latest stable version of Julia. 

### VS Code (optional)

We are going to write and execute code from a graphical interface, generated directly from Julia (see the previous step).

In general, however, it might come in handy to install a professional text editor such as Visual Studio Code.

You can do it by following the download button from [its official website](https://code.visualstudio.com/). 

Once installed, open the `Extensions` view (`Ctrl+Shift+X`), then enter the keyword `julia` and install the extension maintained by the `julialang` organization.

### Opening your first notebook

Download the zip of this project from github, clicking the green button, then unzip the result.
If you installed VS Code, open it and click `File`, then `open folder` and select the project's folder.
Then, click `Terminal` and `new terminal`.

If you did not install VS Code, simply open a new terminal pointing to the project's directory.
For example, on Windows, enter a folder, then `right click` and do `Open in Terminal`.

Now write the command `julia --project=.` and press enter.
You are now *inside* a Julia session.
To download all the necessary dependencies, digit `] instantiate` and press enter.

The process might take a few minutes.

Now write the commands `import Pluto` and then `Pluto.run()`.

You are now ready to go ; )

