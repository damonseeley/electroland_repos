package net.electroland.elvis.manager;

import javax.swing.ImageIcon;
import javax.swing.JButton;

public class ToolBarButton extends JButton {
	
	public ToolBarButton(String imageFileName, String text) {
		super(new ImageIcon(imageFileName));
		setVerticalTextPosition(BOTTOM);
	    setHorizontalTextPosition(CENTER);
		this.setToolTipText(text);
	}

}
