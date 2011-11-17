package net.electroland.installsim.ui;

import java.awt.BorderLayout;
import java.awt.DefaultKeyboardFocusManager;
import java.awt.KeyEventDispatcher;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.KeyStroke;

import net.electroland.installsim.core.ModelGeneric;

public class EISFrame extends JFrame implements KeyEventDispatcher {

	public static EISPanel eISPanel;

	//public static CoopModes mode = new CoopModes();

	// this is here to avoid annoying warning in eclipse
	private static final long serialVersionUID = 1L;
	
	
	JPanel curButtonBar = null;

	public EISFrame(int w, int h, ModelGeneric m) {
		super("Install Simulator");
		//setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);   I think this causes System.exit(0) to be called ASAP
		//so the windowAdpter doesn't have time to clean up 

		DefaultKeyboardFocusManager.getCurrentKeyboardFocusManager().addKeyEventDispatcher(this); // catch key events
		setSize(w, h);

		setupMenus();

		// add buttons to the bottom of the window
		setButtonBar(new EISButtonBar());

		//add the coopPanel to the top of the window
		eISPanel = new EISPanel(w,h,m);
		getContentPane().add(eISPanel, BorderLayout.PAGE_START);

		// Keyboard setup here (and in the class CoopKeyListener)
		addKeyListener(new EISKeyListener());

		addWindowListener(new WindowAdapter() { // setupMenus() for explanation of anonymous classes (this syntax)
			public void windowClosing(WindowEvent event) {
				try { // surround w/try catch block to make sure System.exit(0) gets call no matter what
					//InstallSimMain.killTheads();
				} catch (Exception e) {
					e.printStackTrace();
				}
				try{ // split try/catch so if there is a problem killing threads lights will still die
					//CoopLightsMain.killLights();
				} catch (Exception e) {
					e.printStackTrace();
				}
				System.exit(0);				
			}
		});

		// this puts the menubar in the correct place on macs (at the top of the screen vs at the top of the window) 
		// but won't effect windows.  Its a good habit to call it before you create your frame
		System.setProperty("apple.laf.useScreenMenuBar", "true");
		
		pack(); // pack the layout
		setVisible(true);
	}

	public void setButtonBar(JPanel buttonBar) {
		if(curButtonBar != null) {
			remove(curButtonBar);
		} 
		
		curButtonBar = buttonBar;
		if(curButtonBar != null) {		
			getContentPane().add(curButtonBar, BorderLayout.PAGE_END);			
		}
		pack();
		
	}
	private void setupMenus() {
		JMenuBar menuBar = new JMenuBar();

		// this will serve as a temporary holder to decleare menus and thier items
		// we are going to re-use them
		JMenu menu;
		JMenuItem menuItem;

		// start menu 1
		menu = new JMenu("menu 1");

		menuItem = new JMenuItem("Item 1"); // create an menuItem
		//The next few lines of code are probably going to be the funkiest java syntax you will ever see
		//what you are doing is:
		//  1) defining an ActionListener class, 
		//  2) creating an instance of that newly defined ActionListener class, and
		//  3) passing the newly instantiated ActionListener object into the addActionListenerMethod of menuItime 
		//and you are doing all three operations at one time.  If you are curios the construct is call
		//an anonymous class (because you never give the newly defined class a name) and it saves you from
		//having to create oodles of single use classes (for things like menuitem action listeners)
		menuItem.addActionListener(new ActionListener() { // create an action listener
					public void actionPerformed(ActionEvent e) {
						System.out.println("menu 1 item 1 was invoked");
						//mode.setMode(CoopModes.MODE.INIT);
						setButtonBar(null); // not buttons for this mode
					}
				});
		menu.add(menuItem);

		menuItem = new JMenuItem("Item 2"); // create an menuItem
		menuItem.addActionListener(new ActionListener() { // create an action listener
					public void actionPerformed(ActionEvent e) {
						System.out.println("menu 1 item 2 was invoked");
						//if(mode.getMode() != CoopModes.MODE.MODE1) {
							//mode.setMode(CoopModes.MODE.MODE1);
							//setButtonBar(new CoopButtonBar()); // swap out the button bar for a differnt mode
						//}
					}
				});
		// lets add a shortcut this time
		// its ugly but this is the perfered OS independant way to add a keyboard shortcut
		// you don't have to do anything else of this shortcut to work (IE do not also catch the key with a keyListener)
		// The os might not let you assing a key if it reserved for os functionaly (EG macs don't let you use h becuase it already hides app, or q because it quits)
		menuItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_G, Toolkit
				.getDefaultToolkit().getMenuShortcutKeyMask()));
		menu.add(menuItem);

		menuBar.add(menu); // after the menu is set up add it to the menubar

		// lets create a 2nd menu
		menu = new JMenu("menu 2");
		menuItem = new JMenuItem("menu 2 item 1"); // create an menuItem
		menuItem.addActionListener(new ActionListener() { // create an action listener
					public void actionPerformed(ActionEvent e) {
						System.out.println("menu 2 item 1 was invoked");
						//if(mode.getMode() != CoopModes.MODE.MODE2) {
						//	mode.setMode(CoopModes.MODE.MODE2);
							//setButtonBar(new CoopButtonBar2());// swap out the button bar for a differnt mode
						//}
					}
				});
		menuItem.setAccelerator(KeyStroke.getKeyStroke(KeyEvent.VK_D, Toolkit
				.getDefaultToolkit().getMenuShortcutKeyMask()));
		menu.add(menuItem);

		menuBar.add(menu); // add new menut to the menubar

		setJMenuBar(menuBar); //after the menubar is done add it to the jfame

	}

	  public boolean dispatchKeyEvent(KeyEvent e)  {
		    processKeyEvent(e);
		    return true ;
		    //return false when KeyEvent should not be dispatched to other KeyEventDispatcher
	  }
}