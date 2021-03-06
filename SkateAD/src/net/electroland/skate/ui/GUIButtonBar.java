package net.electroland.skate.ui;

import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import net.electroland.skate.core.*;
import javax.swing.JButton;
import javax.swing.JPanel;



public class GUIButtonBar extends JPanel {
	// added to elim Eclipse warning
	private static final long serialVersionUID = 1L;

	public GUIButtonBar() {
		setLayout(new FlowLayout());
		
		JButton button;
		
		button = new JButton("Add Random Skater");
		// this should look a lot like add actionlisteners to the menuItems
		button.addActionListener(new ActionListener() { // create an action listener
			public void actionPerformed(ActionEvent e) {
				SkateMain.addRandomSkater();
				//CoopLightsMain.killLights();
				//InstallSimMain.trackingComm.outputXforms();
				//System.out.println("button 1 pressed(and released)");
			}
		});
		button.setMnemonic('1'); // respond to alt-b (actualy I'm not sure if its alt or cntr on windows)
		button.setDisplayedMnemonicIndex(7); // uderline the 8th letter of buttons name as a hint
		button.setToolTipText("here is a tool tip just for fun");
		add(button);
		
		button = new JButton("Freeze");
		// this should look a lot like add actionlisteners to the menuItems
		button.addActionListener(new ActionListener() { // create an action listener
			public void actionPerformed(ActionEvent e) {
				SkateMain.freeze = !SkateMain.freeze;
				System.out.println("button 2 pressed(and released)");
			}
		});
		// yet another way to get keyboard shortcuts.
		//button.setMnemonic('2'); // respond to alt-b (actualy I'm not sure if its alt or cntr on windows)
		//button.setDisplayedMnemonicIndex(9); // uderline the 8th letter of buttons name as a hint
		add(button);

		button = new JButton("Button 1.3");
		// this should look a lot like add actionlisteners to the menuItems
		button.addActionListener(new ActionListener() { // create an action listener
			public void actionPerformed(ActionEvent e) {
				//System.out.println("button 3 pressed(and released)");
				//System.out.println(InstallSimConductor.peopleCount() + " people in the space");
			}
		});
		// yet another way to get keyboard shortcuts.
		button.setMnemonic('3'); // respond to alt-b (actualy I'm not sure if its alt or cntr on windows)
		button.setDisplayedMnemonicIndex(9); // uderline the 8th letter of buttons name as a hint
		button.setToolTipText("here is a tool tip just for fun");
		add(button);

	}

}
