package net.electroland.elvis.regionManager;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JToolBar;

public class CreatorToolBar extends JToolBar {
	
	ImagePanel imagePanel;
	
	public CreatorToolBar(ImagePanel ip) {
		imagePanel = ip;
		JButton button = new ToolBarButton("depends/images/cursor.gif", "draw");
		button.addActionListener(new DrawListener());
		add(button);

		
		button = new ToolBarButton("depends/images/cursorEdit.gif", "point edit");
		button.addActionListener(new EditListener());
		add(button);
		
		
		this.setOrientation(JToolBar.VERTICAL);
		setFloatable(false);
		
	}
	
	
	
	
	
	public class DrawListener implements ActionListener {
		public void actionPerformed(ActionEvent e) {
//			imagePanel.changeMode(ImagePanel.Mode.DRAWING);
		}
	}

	public class EditListener implements ActionListener {
		public void actionPerformed(ActionEvent e) {
//			imagePanel.changeMode(ImagePanel.Mode.EDITING);
		}
	}

}
