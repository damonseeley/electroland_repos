package net.electroland.faces;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

import javax.swing.JCheckBox;
import javax.swing.JPanel;
import javax.swing.JRadioButton;

@SuppressWarnings("serial")
public class BuildingJPanel extends JPanel implements MouseListener, ActionListener{

	private Image physicsModel = null;

	// yuck.  these defaults should check with LAFACEApp.java on instantiation.
	private boolean isManual = false;
	private boolean sendToBuilding = true;

	// Each controller represents a lighting universe.  E.g., a single
	// recipient of an ArtNet packet.
	private LightController[] controllers;
	// Each light is a pixel.
	private Light[] lights;

	public BuildingJPanel(boolean isDoubleBuffered, LightController[] controllers){
		super(isDoubleBuffered);
		this.controllers = controllers;
		lights = Util.getAllLights(controllers);
	}

	@Override
	protected void paintComponent(Graphics g) {
		
		super.paintComponent(g);

		if (physicsModel == null){
			/* paint the background */
			g.setColor(Color.BLACK);
			g.fillRect(0, 0, this.getWidth(), this.getHeight());
			
		}else{
			/* paint the physics model, if provided */
			
		}
		
		/* paint the lights, white for on, white border/black interior for off */
		for (int i = 0; i < lights.length; i++){
			// if we are hurting on performance, change this to:
			// g.setColor(brightness[i].color == 0 ? Color.BLACK : Color.WHITE);
			g.setColor(new Color(lights[i].brightness, lights[i].brightness, lights[i].brightness));
			g.fillRect(lights[i].lightbox.x, lights[i].lightbox.y,
					lights[i].lightbox.width, lights[i].lightbox.height);
			g.setColor(Color.WHITE);
			g.drawRect(lights[i].lightbox.x, lights[i].lightbox.y,
					lights[i].lightbox.width, lights[i].lightbox.height);
		}		

		// send to the building
		if (sendToBuilding){
			for (int i = 0; i < controllers.length; i++){
				controllers[i].send();
			}
		}

	}

	public void actionPerformed(ActionEvent e) {
		if (e.getSource() instanceof JRadioButton){
			String mode = ((JRadioButton)e.getSource()).getText();
			this.isManual = mode.equalsIgnoreCase(LAFACEApp.MANUAL_MODE);
			
			// between modes, disable all lights.
			for (int i = 0; i < lights.length; i++){
				lights[i].brightness = 0;
			}
			repaint();
		}else if (e.getSource() instanceof JCheckBox){
			String mode = ((JCheckBox)e.getSource()).getText();
			if (mode.equalsIgnoreCase(LAFACEApp.SEND_TO_BUILDING)){
				this.sendToBuilding = ((JCheckBox)e.getSource()).isSelected();				
			}
		}
	}	
	
	public void mouseReleased(MouseEvent e) {		

		if (isManual){		
			int x = e.getX();
			int y = e.getY();
			
			for (int i = 0; i < lights.length; i++){
				if (lights[i].contains(x, y)){
					lights[i].brightness = (lights[i].brightness == 0 ? 255 : 0);
					System.out.println("Hit: " + lights[i]);
				}
			}
			repaint();
		}
	}

	public void setManualMode(boolean isManual){
		this.isManual = isManual;
	}

	public void mouseClicked(MouseEvent e) {}

	public void mouseEntered(MouseEvent e) {}

	public void mouseExited(MouseEvent e) {}

	public void mousePressed(MouseEvent e) {}
}