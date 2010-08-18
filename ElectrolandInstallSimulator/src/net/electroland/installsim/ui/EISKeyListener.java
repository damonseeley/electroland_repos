package net.electroland.installsim.ui;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.Random;


public class EISKeyListener implements KeyListener {
	
	public void keyTyped(KeyEvent e) {		
	}

	public void keyPressed(KeyEvent e) {		
	}
	
	public void keyReleased(KeyEvent e) {
		if(e.isAltDown()) {
			switch(e.getKeyCode()) {
			// use getKeyCode() and not getKeyChar 
			// because getKeyChar() only works with non-modified key downs (since shifts, alts, etc modify the keyChar but not the keyCode) 
			case KeyEvent.VK_R:
				System.out.println("got a alt-r");
				break;
			case KeyEvent.VK_Z:
				System.out.println("got a alt-z");
				break;
			case KeyEvent.VK_X:
				System.out.println("got a alt-x");
				break;
			case KeyEvent.VK_C:
				System.out.println("got a alt-c");
				break;
			case KeyEvent.VK_V:
				System.out.println("got a alt-v");
				break;
			case KeyEvent.VK_B:
				System.out.println("got a alt-b");
				break;
			case KeyEvent.VK_N:
				System.out.println("got a alt-n");
				break;
			case KeyEvent.VK_M:
				System.out.println("got a alt-m");
				break;
			case KeyEvent.VK_COMMA:
				System.out.println("got a alt-,");
				break;
				
			}
		} else if(e.isControlDown()) {  
			// you can also check for isShiftDown, isMetaDown (usually means esc), 
			// or any combination with booleans eg (e.isControlDown() && is.shiftDown())
			switch(e.getKeyCode()) {
			case KeyEvent.VK_A:
				System.out.println("got a ctrl-a");
				break;
			case KeyEvent.VK_B:
				System.out.println("got a ctrl-b");
				break;
			case KeyEvent.VK_C:
				System.out.println("got a ctrl-c");
				break;			
			}
		} else {
			float offsetinc = 2.54f;
			float scaleinc = .01f;
			switch(e.getKeyCode()) {
			case KeyEvent.VK_Z:
				System.out.println("x offset modded by " + -1*offsetinc);
				//InstallSimMain.trackingComm.updateXTransformOffset(-1*offsetinc);
				break;
			case KeyEvent.VK_X:
				System.out.println("x offset modded by " + offsetinc);
				//InstallSimMain.trackingComm.updateXTransformOffset(offsetinc);
				break;
			case KeyEvent.VK_C:
				System.out.println("y offset modded by " + -1*offsetinc);
				//InstallSimMain.trackingComm.updateYTransformOffset(-1*offsetinc);
				break;
			case KeyEvent.VK_V:
				System.out.println("y offset modded by " + offsetinc);
				//InstallSimMain.trackingComm.updateYTransformOffset(offsetinc);
				break;
			case KeyEvent.VK_B:
				System.out.println("x scale modded by " + -1*scaleinc);
				//InstallSimMain.trackingComm.updateXTransformScale(-1*scaleinc);
				break;
			case KeyEvent.VK_N:
				System.out.println("x scale modded by " + scaleinc);
				//InstallSimMain.trackingComm.updateXTransformScale(scaleinc);
				break;
			case KeyEvent.VK_M:
				System.out.println("y scale modded by " + -1*scaleinc);
				//InstallSimMain.trackingComm.updateYTransformScale(-1*scaleinc);
				break;
			case KeyEvent.VK_COMMA:
				System.out.println("yx scale modded by " + scaleinc);
				//InstallSimMain.trackingComm.updateYTransformScale(scaleinc);
				break;
				
				//RESTART STUFF
			case KeyEvent.VK_R:
				System.out.println("setting lights for restart");
				//InstallSimMain.executeRestart();
				break;
				
				//QUIT
			case KeyEvent.VK_Q:
//				System.out.println("shutting down");
//				CoopLightsMain.shutdown();
				break;
				
				//AMBIENT TESTS
			case KeyEvent.VK_T:
				System.out.println("thermometer added");

				break;
				
			case KeyEvent.VK_I:
				System.out.println("thermometer added");

				break;
				
			case KeyEvent.VK_U:
				System.out.println("mega fill added");

				break;
				
			case KeyEvent.VK_E:
				//System.out.println("add easter eggs");

				break;
				
			case KeyEvent.VK_Y:
				System.out.println("zipper added");

				break;
				
			case KeyEvent.VK_O:
				System.out.println("stage all on added");
				
				break;
				
			case KeyEvent.VK_K:
				System.out.println("stage all on added");

				break;
				
				//OTHER TESTS
			case KeyEvent.VK_D:
				//System.out.println("People Dists");

				break;
				
			case KeyEvent.VK_S:

				break;
				
				//create the test people
			case KeyEvent.VK_P:
				//InstallSimMain.createTestPerson();
				break;
				
				
				
			case KeyEvent.VK_SPACE:
				//if (InstallSimMain.SHOWUI){
					//InstallSimMain.SHOWUI = false;
				//} else {
					//InstallSimMain.SHOWUI = true;
				//}
				break;
				


			}
			
		}
	}
	
}
