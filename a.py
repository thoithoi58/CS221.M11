from tkinter import *
root=Tk()
def retrieve_input():
    inputValue=textBox.get("1.0",END).splitlines()
    for i in inputValue:
        print(i)
    # print(inputValue)

textBox=Text(root, height=2, width=10)
textBox.insert(INSERT, "AAAAA")
textBox.pack()
buttonCommit=Button(root, height=1, width=10, text="Commit", 
                    command=lambda: retrieve_input())
#command=lambda: retrieve_input() >>> just means do this when i press the button
buttonCommit.pack()

mainloop()