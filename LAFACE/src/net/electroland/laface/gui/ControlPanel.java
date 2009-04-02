package net.electroland.laface.gui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Event;
import java.awt.Label;
import java.awt.Scrollbar;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;

import net.electroland.laface.core.LAFACEMain;
import net.electroland.laface.shows.WaveShow;
import net.electroland.laface.sprites.Wave;
import net.electroland.lighting.detector.animation.Completable;
import net.miginfocom.swing.MigLayout;

/**
 * Contains control widgets for adjusting show parameters.
 * @author asiegel
 */

@SuppressWarnings("serial")
public class ControlPanel extends JPanel implements ActionListener, ChangeListener, ListSelectionListener{
	
	private LAFACEMain main;
	private Scrollbar dampingSlider, fpuSlider, yoffsetSlider, dxSlider, cSlider, brightnessSlider, alphaSlider;
	private JButton resetWaveButton;
	private DefaultListModel waveListModel;
	private JList waveList;
	private int currentWaveID = -1;
	private int width, height;

	public ControlPanel(LAFACEMain main){
		this.main = main;
		width = 1048;
		height = 150;
		setMinimumSize(new Dimension(width,height));
		//setBackground(Color.black);
		//setForeground(Color.white);
		setLayout(new MigLayout("insets 0 0 0 0"));
		
		JTabbedPane tabbedPane = new JTabbedPane();
		//tabbedPane.setForeground(Color.white);
		//tabbedPane.setBackground(Color.black);

		tabbedPane.addTab("Draw Test", makeDrawTestPanel());
		tabbedPane.addTab("Trace Test", makeTraceTestPanel());
		tabbedPane.addTab("Wave Show", makeWaveShowPanel());
		
		tabbedPane.setMinimumSize(new Dimension(width/2,height));
		tabbedPane.addChangeListener(this);
		add(tabbedPane);
		tabbedPane.setTabLayoutPolicy(JTabbedPane.SCROLL_TAB_LAYOUT);
		
		add(makeDisplayModePanel());

	}
	
	public JComponent makeDrawTestPanel(){
		JPanel panel = new JPanel(false);
        panel.setLayout(new MigLayout(""));
		panel.setMinimumSize(new Dimension(width/2,height));
		return panel;
	}
	
	public JComponent makeTraceTestPanel(){
		JPanel panel = new JPanel(false);
        panel.setLayout(new MigLayout(""));
		panel.setMinimumSize(new Dimension(width/2,height));
		return panel;
	}
	
	public JComponent makeWaveShowPanel(){
		JPanel panel = new JPanel(false);
        panel.setLayout(new MigLayout(""));
		panel.setMinimumSize(new Dimension(width/2,height));
		
		waveListModel = new DefaultListModel();
		waveList = new JList(waveListModel);
		waveList.setMinimumSize(new Dimension(100,150));
		waveList.addListSelectionListener(this);
		panel.add(waveList, "west");

		
		// slider for adjusting damping value
		panel.add(new Label("Damping", Label.RIGHT));
		dampingSlider = new Scrollbar(Scrollbar.HORIZONTAL, 0, 1, 0, 100);
		dampingSlider.setForeground(Color.black);
		dampingSlider.setBackground(Color.white);
		dampingSlider.setMinimumSize(new Dimension(100, 16));
		panel.add(dampingSlider, "wrap");
		
		// slider for adjusting nonlinearity value
		panel.add(new Label("Nonlinearity", Label.RIGHT));
		fpuSlider = new Scrollbar(Scrollbar.HORIZONTAL, 0, 1, 0, 100);
		fpuSlider.setForeground(Color.black);
		fpuSlider.setBackground(Color.white);
		fpuSlider.setMinimumSize(new Dimension(100, 16));
		panel.add(fpuSlider, "wrap");
		
		// slider for adjusting y offset of wave surface
		panel.add(new Label("Y-Offset", Label.RIGHT));
		yoffsetSlider = new Scrollbar(Scrollbar.HORIZONTAL, 60, 1, 0, 100);
		yoffsetSlider.setForeground(Color.black);
		yoffsetSlider.setBackground(Color.white);
		yoffsetSlider.setMinimumSize(new Dimension(100, 16));
		panel.add(yoffsetSlider, "wrap");
		
		// slider for adjusting mysterious dx value (relates to horizontal wave speed)
		panel.add(new Label("DX value", Label.RIGHT));
		dxSlider = new Scrollbar(Scrollbar.HORIZONTAL, 2, 1, 0, 100);
		dxSlider.setForeground(Color.black);
		dxSlider.setBackground(Color.white);
		dxSlider.setMinimumSize(new Dimension(100, 16));
		panel.add(dxSlider, "wrap");
		
		// slider for adjusting mysterious c value (relates to wave speed)
		panel.add(new Label("C value", Label.RIGHT));
		cSlider = new Scrollbar(Scrollbar.HORIZONTAL, 12, 1, 0, 100);
		cSlider.setForeground(Color.black);
		cSlider.setBackground(Color.white);
		cSlider.setMinimumSize(new Dimension(100, 16));
		panel.add(cSlider, "wrap");
		
		// slider for adjusting brightness
		panel.add(new Label("Brightness", Label.RIGHT));
		brightnessSlider = new Scrollbar(Scrollbar.HORIZONTAL, 255, 1, 0, 255);
		brightnessSlider.setForeground(Color.black);
		brightnessSlider.setBackground(Color.white);
		brightnessSlider.setMinimumSize(new Dimension(100, 16));
		panel.add(brightnessSlider, "wrap");
		
		// slider for adjusting alpha
		panel.add(new Label("Alpha", Label.RIGHT));
		alphaSlider = new Scrollbar(Scrollbar.HORIZONTAL, 255, 1, 0, 255);
		alphaSlider.setForeground(Color.black);
		alphaSlider.setBackground(Color.white);
		alphaSlider.setMinimumSize(new Dimension(100, 16));
		panel.add(alphaSlider);
		
		// button for resetting wave after it goes haywire
		resetWaveButton = new JButton("Reset Wave");
		resetWaveButton.addActionListener(this);
		panel.add(resetWaveButton);
		
		
		return panel;
	}
	
	public JComponent makeDisplayModePanel(){
		JPanel panel = new JPanel(false);
        panel.setLayout(new MigLayout(""));
		panel.setMinimumSize(new Dimension(width/2,height));
		panel.add(new Label("Display Mode:"), "wrap");
		JComboBox displayModeList = new JComboBox(new String[] {"Raster","Raster + Detectors", "Detector Values"});
		displayModeList.setSelectedIndex(1);
		displayModeList.addActionListener(this);
		panel.add(displayModeList);
		return panel;
	}


	public void actionPerformed(ActionEvent e) {
		//System.out.println(e.getActionCommand());
		if(e.getActionCommand().equals("Reset Wave")){
			if(currentWaveID != -1){					// if a wave sprite has been selected in the wave list...
				Completable a  = main.getCurrentAnimation();
				if(a instanceof WaveShow){			// confirm show is WaveShow
					ConcurrentHashMap<Integer, Wave> waves = ((WaveShow) a).getWaves();
					Wave wave = waves.get(currentWaveID);
					wave.reset();
				}
			}
		} else if(e.getActionCommand().equals("comboBoxChanged")){
			JComboBox cb = (JComboBox)e.getSource();
		    if((String)cb.getSelectedItem() == "Raster"){
		    	main.rasterPanel.setDisplayMode(0);
		    } else if((String)cb.getSelectedItem() == "Raster + Detectors"){
		    	main.rasterPanel.setDisplayMode(1);
		    } else if((String)cb.getSelectedItem() == "Detector Values"){
		    	main.rasterPanel.setDisplayMode(2);
		    }
		}
	}
	
	public boolean handleEvent(Event e){
		//System.out.println(e);
		// TODO MUST specify a Wave sprite instance to modify
		if(e.target instanceof Scrollbar){
			if(currentWaveID != -1){	// if a wave sprite has been selected in the wave list...
				Completable a  = main.getCurrentAnimation();
				if(a instanceof WaveShow){			// confirm show is WaveShow
					ConcurrentHashMap<Integer, Wave> waves = ((WaveShow) a).getWaves();
					Wave wave = waves.get(currentWaveID);
					if(e.target.equals(dampingSlider)){
						wave.setDamping(dampingSlider.getValue()/100.0);
					} else if (e.target.equals(fpuSlider)){
						wave.setNonlinearity(fpuSlider.getValue()/100.0);
					} else if (e.target.equals(yoffsetSlider)){
						wave.setYoffset(yoffsetSlider.getValue()/100.0);
					} else if (e.target.equals(dxSlider)){
						wave.setDX(dxSlider.getValue()/100.0);
					} else if (e.target.equals(cSlider)){
						wave.setC(cSlider.getValue()/100.0);
					} else if (e.target.equals(brightnessSlider)){
						wave.setBrightness(brightnessSlider.getValue());
					} else if (e.target.equals(alphaSlider)){
						wave.setAlpha(alphaSlider.getValue());
					}
				}
			}
		}
		return false;
	}

	public void stateChanged(ChangeEvent e) {
		//System.out.println(((JTabbedPane)e.getSource()).getSelectedIndex());
		if(((JTabbedPane)e.getSource()).getSelectedIndex() == 0){
			// TODO switch to Draw Test
		} else if(((JTabbedPane)e.getSource()).getSelectedIndex() == 1){
			// TODO switch to Trace Test
		} else if(((JTabbedPane)e.getSource()).getSelectedIndex() == 2){
			// TODO switch to Wave Show
			Completable a  = main.getCurrentAnimation();
			if(a instanceof WaveShow){	// confirm show has been switched over
				waveListModel.clear();
				ConcurrentHashMap<Integer, Wave> waves = ((WaveShow) a).getWaves();
				Iterator<Wave> iter = waves.values().iterator();
				int counter = 0;
				while(iter.hasNext()){
					Wave wave = iter.next();
					waveListModel.addElement("wave "+counter);
					counter++;
				}
			}
		}
	}

	public void valueChanged(ListSelectionEvent e) {
		if (e.getValueIsAdjusting() == false) {		// done adjusting
	        if(waveList.getSelectedIndex() >= 0){		// if mouse event
	        	// TODO populate controls with current waves properties
	        	currentWaveID = waveList.getSelectedIndex();
	        	Completable a  = main.getCurrentAnimation();
				if(a instanceof WaveShow){			// confirm show is WaveShow
					ConcurrentHashMap<Integer, Wave> waves = ((WaveShow) a).getWaves();
					Wave wave = waves.get(currentWaveID);
					dampingSlider.setValue((int)(wave.getDamping()*100));
					fpuSlider.setValue((int)(wave.getNonlinearity()*100));
					yoffsetSlider.setValue((int)(wave.getYoffset()*100));
					dxSlider.setValue((int)(wave.getDX()*100));
					cSlider.setValue((int)(wave.getC()*100));
					brightnessSlider.setValue(wave.getBrightness());
					alphaSlider.setValue(wave.getAlpha());
				}
	        }
	    }

	}
	
}
