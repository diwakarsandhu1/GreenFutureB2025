// main.js

const { app, BrowserWindow, Menu } = require('electron');
const path = require('path');

// Create the Electron window
function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
    }
  });
  
  const isDev = !app.isPackaged;

  if (isDev) {
    win.loadURL("http://localhost:3000");
  } else {
    win.loadFile(path.join(__dirname, "index.html"));
  }



const menu = Menu.buildFromTemplate([
  {
    label: 'Menu',
    submenu: [
      {
        label: 'Open Dev Tools',
        click() {
          win.webContents.openDevTools();
        },
      },
      {
        label: 'Reload',
        role: 'reload',
      },
      {
        label: 'Quit',
        accelerator: 'CmdOrCtrl+Q',
        click() {
          app.quit();
        },
      },
    ],
  },
]);

Menu.setApplicationMenu(menu);


}

// This will be called once Electron is ready
app.whenReady().then(() => {
  createWindow();

  // Handle macOS app behavior (keep the app open when no windows are open)
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Quit the app when all windows are closed (except on macOS)
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});