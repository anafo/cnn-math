import matplotlib.pyplot as plt
plt.title('Learning Curves')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(history['loss'], label='train')
plt.plot(history['val_loss'], label='val')
plt.legend()
plt.show()