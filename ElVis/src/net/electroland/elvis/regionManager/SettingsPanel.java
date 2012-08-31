package net.electroland.elvis.regionManager;

import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

public class SettingsPanel extends JPanel implements ChangeListener {
	boolean isSelected = false;
	
	JTabbedPane parentPane;
	
	public SettingsPanel(JTabbedPane parentPane) {
		this.parentPane = parentPane;
		parentPane.addChangeListener(this);
	}
	
	
	public void panelSelected() {
	}
	
	public void panelDeselected() {		
	}
	public void stateChanged(ChangeEvent e) {
		if(parentPane.getSelectedComponent() == this) {
			isSelected = true;
			panelSelected();
		} else {
			if(isSelected) {
				isSelected = false;;
				panelDeselected();
			}
		}
		
	}

}
