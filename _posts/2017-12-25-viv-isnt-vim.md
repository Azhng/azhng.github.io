---
layout: post
title:  "Viv isn't Vim"
date:   2017-12-25 23:04:45 -0400
category: professional
---

### Viv Isn't Vim

Viv is a lightweight C++ implementation of vim. Viv implements most of vim's core feature including most of the navigation commands and editing commands, command multipliers, undo/redo, macros, syntax highlight for C++, bracket matching, and file I/O.

Here is the [link][viv-github] to the github repo. However, do note that due to university policy, the source code of this project is not publically available. Hence that repo only contains pre-built binaries for Linux and macOS. But feel free to shoot me an email and I will be more than happy to send you a copy of the source code, provided that you are not going to submit it as your own project of course. 


## How did it start ? 
Well, it all started with [Lawrence][lpan-site]. Lawrence is a really good friend of mine and also an absolute legend. Back when I was still working at TD Lab last summer, he told me he was working on a vim clone, [viw][viw-github] (Vi Worsen), written completely from scratch using C. According to him, he spent entire reading week break working on that project and nothing else. However, that project did help him get his first internship at Universe. Though eventually I only contributed 8 lines of code to Lawrence's viw to fix a compatibility issue on macOS, the architecture choice of viv was heavily influenced by viw.

Few months later, when I finished my internship at TD Lab and went back to school for my 2A study term. Quite coincidentally, the final project for one of the courses that I was taking (CS 246E - Object Oriented Software Development) was writing a clone of vim in three weeks. 

## The Architecture
The overall architecture of viv is loosely based on the classic MVC architecture. 

The Model layer consists of two `State` objects, a `PaintableState` and a `LogicState`. Intuitively, the `PaintableState` object is used to represent everything that's viewable within the applicaiton, including the current state of the document, syntax highlight infomation, cursor information and viewport information. Conversely, the `LogicState` object is used to record all information that is not directly viewable. 

The View layer is completely stateless. It takes a constant reference of a `PaintableState` object and renders it using ncurses library. 

The Controller layer is the one my partner and I spent most of my time debating, because eventually we need to be able to support undo/redo and macro operations. That means viv need to be able to record everything single mutations that were applied to the `PaintableState` object, and it also need to be able to reverse all the mutations. Naively, our original solution was to make the `PaintableState` object completely immutable, every command that can be undone will generate a new `PaintableState` object to replace the old one, and the old `PaintableState` object will be stored onto a stack. This is probably the easiest viable solution to implement undo/redo, however this solution is probably going to blow up the memory if an large file is being opened. (Say like 20 MB). Moreover, it makes implementing macro very tricky. Since when user undoes a macro, all operations captured by the macro will be undone. That means speical data structures need to be created just to handle this one single feature. This also creates a strong coupling between the Controler layer and the Model layer.

Our second solution was to store a stack of diff objects that captures the difference between `PaintableState` objects before and after mutation. This method is way more memory friendly compared to the first method. However, it is still difficult to implement macros using this solution due to the same reasons for the first solution. 

Our final solution is what we eventually ended up using. Instead of directly apply operation to the Model layer, the Controller generates an `Action` object. Each `Action` object corresponds to a vim command. All `Action` objects define `Action::execute` method. For `Action` that can be undone, the `Action` object implements an additional `Action::undo` method. Then it becomes extremely trivial to implement undo/redo and macros. Following pseudocode illustrate the concept of this strategy.

```
void appLoop() {
    int command = Keyboard::getKeystroke();
    Action* action = ActionFactory::buildAction(command); 
    if (action->isUndoable) {
        store_action(action);
    }
    action->execute(&paintable_state);
    View::render(paintable_state);
}
```

Observe that in above code snippet, a lot of method are implemented as static method. This is because most of the objects in viv are actually stateless and there is no need to create a new object (though I could have done it better by just implement them using functions in a nested namespace). This is to minimize the coupling between different modules. However, in the actual code we made a heavy use of `unique_ptr` to manage the memory for us. The advantage of this strategy is it really decouples the main application logic from the logic of each individual action. There are times that we need to implement some hacks to get some `Action` working, but thanks to this architecture all the ugliness of those dirty hacks are self contained within the `Action` class itself and will not affect the code quality in other places. Neverthelss, this strategy is not perfect either. There exists a lot of boilerplate code in this project since all `Action` objects inherits from a single base class. If I remember correctly, we ended up with 121 files and about 50 to 60 classes (the entire project is about ~8800 lines). This becomes extremely troublesome when we need to recompile the entire project, roughly it takes about 40 to 50 seconds to compile everything with all the test cases.


## C vs C++ 
Compared to Lawrence's viw, viv does have more features. Though this is mainly due to the fact that we had three weeks and two people to work on this project. Also we were working on a higher abstraction level since we were using C++ instead of C. In Lawrence viw, he spent most of his time implement all kinds of doubly linked list to create abstractions for window and documents. Whereas in C++, the standard library comes with various nice STL containers which we made very heavy use of, and also smart pointers which makes memory management extremely easy. I am not saying C++ is a superior language than C, but it is clear that C++ does have a clear advantage than C when it comes to application development.

## What's next
I just finished my 2A study term (barely survived) and will start my internship at deeplearni.ng this next year. I thinking about writing another post that summarize everything that happened to me this year, it's probably just going to be a generic post and not really tech related. And also I'm planning for my next big side project at the moment, I'll probably post something when I have something that's working.


[lpan-site]: http://lpan.io
[viv-github]: https://github.com/Azhng/viv
[viw-github]: https://github.com/lpan/viw
