#!/usr/bin/env python
# coding: utf-8

# In[29]:


import re
import matplotlib.pyplot as plt

# Initialize lists to store loss values
loss_values = []
epoch_numbers = []

# Open and parse the file
with open('training.txt', 'r') as f:
    for line in f:
        # Extract epoch number and loss value using regular expressions
        epoch_match = re.search(r'Epoch: \[(\d+)\]', line)
        loss_match = re.search(r'Loss (\d+\.\d+)', line)
        
        if epoch_match and loss_match:
            epoch = int(epoch_match.group(1))
            loss = float(loss_match.group(1))
            
            epoch_numbers.append(epoch)
            loss_values.append(loss)

# Print values to verify
print("Epochs:", epoch_numbers)
print("Loss values:", loss_values)


# In[30]:


# Plot the loss values
plt.figure(figsize=(10, 6))
plt.plot(epoch_numbers, loss_values, label="Training Loss", color='blue', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()
#plt.grid()
plt.show()


# In[34]:


import re
import matplotlib.pyplot as plt

# Initialize lists to store loss and accuracy values
loss_values = []
accuracy_values = []
epoch_numbers = []

# Open and parse the file
with open('training.txt', 'r') as f:
    for line in f:
        # Extract epoch number, loss, and accuracy values using regular expressions
        epoch_match = re.search(r'Epoch: \[(\d+)\]', line)
        loss_match = re.search(r'Loss (\d+\.\d+)', line)
        accuracy_match = re.search(r'Prec@1 (\d+\.\d+)', line)
        
        if epoch_match and loss_match and accuracy_match:
            epoch = int(epoch_match.group(1))
            loss = float(loss_match.group(1))
            accuracy = float(accuracy_match.group(1))
            
            epoch_numbers.append(epoch)
            loss_values.append(loss)
            accuracy_values.append(accuracy)

# Print values to verify
print("Epochs:", epoch_numbers)
print("Loss values:", loss_values)
print("Accuracy values:", accuracy_values)


# In[32]:


fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot loss on the first y-axis
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss", color="blue")
#ax1.plot(epoch_numbers, loss_values, label="Training Loss", color="blue", marker="o")
ax1.tick_params(axis="y", labelcolor="blue")

# Create a second y-axis to plot accuracy
ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy (%)", color="green")
ax2.plot(epoch_numbers, accuracy_values, label="Training Accuracy", color="green", marker="o")
ax2.tick_params(axis="y", labelcolor="green")

fig.suptitle("Training Loss and Accuracy per Epoch")
fig.tight_layout()
plt.grid()
plt.show()


# In[35]:


import re
import matplotlib.pyplot as plt

# Initialize lists to store accuracy values and epochs
accuracy_values = []
epoch_numbers = []

# Open and parse the file
with open('training.txt', 'r') as f:
    for line in f:
        # Extract epoch number and accuracy using regular expressions
        epoch_match = re.search(r'Epoch: \[(\d+)\]', line)
        accuracy_match = re.search(r'Prec@1 (\d+\.\d+)', line)
        
        if epoch_match and accuracy_match:
            epoch = int(epoch_match.group(1))
            accuracy = float(accuracy_match.group(1))
            
            epoch_numbers.append(epoch)
            accuracy_values.append(accuracy)

# Plot accuracy values
plt.figure(figsize=(10,6))
plt.plot(epoch_numbers, accuracy_values, label="Training Accuracy", color='green', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy per Epoch")
plt.legend()
#plt.grid()
plt.show()


# In[15]:


get_ipython().system('pip freeze > requirements.txt')


# In[ ]:




