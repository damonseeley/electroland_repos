package net.electroland.laface.gui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Event;
import java.awt.Label;
import java.awt.Scrollbar;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JPanel;

import net.electroland.laface.core.LAFACEMain;
import net.miginfocom.swing.MigLayout;

/**
 * Contains control widgets for adjusting show parameters.
 * @author asiegel
 */

@SuppressWarnings("serial")
public class ControlPanel extends JPanel implements ActionListener{
	
	LAFACEMain main;
	Scrollbar dampingSlider, fpuSlider, yoffsetSlider;

	public ControlPanel(LAFACEMain main){
		this.main = main;
		setMinimumSize(new Dimension(1048,133));
		setBackground(Color.black);
		setForeground(Color.white);
		setLayout(new MigLayout("insets 0 0 0 0"));
		
		// slider for adjusting damping value
		add(new Label("Damping", Label.RIGHT));
		dampingSlider = new Scrollbar(Scrollbar.HORIZONTAL, 0, 1, 0, 100);
		dampingSlider.setForeground(Color.black);
		dampingSlider.setBackground(Color.white);
		dampingSlider.setMinimumSize(new Dimension(100, 16));
		add(dampingSlider, "wrap");
		
		// slider for adjusting nonlinearity value
		add(new Label("Nonlinearity", Label.RIGHT));
		fpuSlider = new Scrollbar(Scrollbar.HORIZONTAL, 0, 1, 0, 100);
		fpuSlider.setForeground(Color.black);
		fpuSlider.setBackground(Color.white);
		fpuSlider.setMinimumSize(new Dimension(100, 16));
		add(fpuSlider, "wrap");
		
		// slider for adjusting y offset of wave surface
		add(new Label("Y-Offset", Label.RIGHT));
		yoffsetSlider = new Scrollbar(Scrollbar.HORIZONTAL, 60, 1, 0, 100);
		yoffsetSlider.setForeground(Color.black);
		yoffsetSlider.setBackground(Color.white);
		yoffsetSlider.setMinimumSize(new Dimension(100, 16));
		add(yoffsetSlider, "wrap");
	}

	public void actionPerformed(ActionEvent e) {
		System.out.println(e.getActionCommand());
	}
	
	public boolean handleEvent(Event e){
		//System.out.println(e);
		if(e.target instanceof Scrollbar){
			if(e.target.equals(dampingSlider)){
				//System.out.println(dampingSlider.getValue());
				ActionEvent event = new ActionEvent(dampingSlider, ActionEvent.ACTION_PERFORMED, "damping:"+String.valueOf(dampingSlider.getValue()));
				main.actionPerformed(event);
			} else if (e.target.equals(fpuSlider)){
				//System.out.println(fpuSlider.getValue());
				ActionEvent event = new ActionEvent(fpuSlider, ActionEvent.ACTION_PERFORMED, "nonlinearity:"+String.valueOf(fpuSlider.getValue()));
				main.actionPerformed(event);
			} else if (e.target.equals(yoffsetSlider)){
				//System.out.println(yoffsetSlider.getValue());
				ActionEvent event = new ActionEvent(yoffsetSlider, ActionEvent.ACTION_PERFORMED, "yoffset:"+String.valueOf(yoffsetSlider.getValue()));
				main.actionPerformed(event);
			}
		}
		return false;
	}
	
}
