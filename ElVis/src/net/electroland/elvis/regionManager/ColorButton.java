package net.electroland.elvis.regionManager;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;

import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JColorChooser;

public class ColorButton extends JButton implements ActionListener {
	Colorable colorable;

	int width;
	int hieght;
	

	public ColorButton(int w, int h, Colorable colorable) {
		setSize(w, h);
		
		width = w;
		hieght = h;
		
		addActionListener(this);
		setColorable(colorable);
	}

	public void setColorable(Colorable colorable) {
		this.colorable = colorable;
		if(colorable != null) {
			setColor(colorable.getColor());
		}

	}
	public void setColor(Color c) {
		if(colorable != null) {
			colorable.setColor(c);
		}
		BufferedImage bi = new BufferedImage(width, hieght, BufferedImage.TYPE_INT_RGB);
		Graphics2D g = bi.createGraphics();
		g.setColor(c);
		g.fillRect(0, 0, width, hieght);
		setIcon(new ImageIcon(bi));
		repaint();
	}

	public void actionPerformed(ActionEvent e) {
		Color newColor = JColorChooser.showDialog(
				this,
				"Color...",
				colorable.getColor());
		if(newColor != null) {
			setColor(newColor);
		}
	}



}
