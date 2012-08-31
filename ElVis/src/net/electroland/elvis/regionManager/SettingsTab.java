package net.electroland.elvis.regionManager;

import java.awt.Dimension;

import javax.swing.JPanel;
import javax.swing.JTabbedPane;

public class SettingsTab extends JTabbedPane  {
	
		
	public SettingsTab() {
		

		JPanel panel = new GlobalSettingsPanelMig().build();
		panel.setSize(300, -1);
		addTab("global", panel);
		
		
		panel = new RegionSettingsPanelMig().build();
		panel.setSize(300, -1);
		addTab("region", panel);

		
		this.setPreferredSize(new Dimension(300, 900));
		
		
	}

	
	


}
