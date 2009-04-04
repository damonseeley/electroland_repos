package net.electroland.laface.gui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Event;
import java.awt.Label;
import java.awt.Scrollbar;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import javax.swing.DefaultListModel;
import javax.swing.JButton;
import javax.swing.JCheckBox;
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
public class ControlPanel extends JPanel implements ActionListener, ChangeListener, ListSelectionListener, ItemListener{
	
	private LAFACEMain main;
	private Scrollbar dampingSlider, fpuSlider, yoffsetSlider, dxSlider, cSlider, brightnessSlider, alphaSlider;
	private Scrollbar traceSpeedSlider;
	private JButton resetWaveButton, saveWavesButton, clearDrawTestButton;
	private JCheckBox tintBlueButton;
	private DefaultListModel waveListModel;
	private JList waveList;
	private int currentWaveID = -1;
	private int width, height;

	public ControlPanel(LAFACEMain main){
		this.main = main;
		width = 1048;
		height = 80;
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
		
		tabbedPane.setMinimumSize(new Dimension((width/4)*3,height));
		tabbedPane.addChangeListener(this);
		add(tabbedPane, "west");
		tabbedPane.setTabLayoutPolicy(JTabbedPane.SCROLL_TAB_LAYOUT);
		
		add(makeDisplayModePanel(), "west");

	}
	
	public JComponent makeDrawTestPanel(){
		JPanel panel = new JPanel(false);
        panel.setLayout(new MigLayout(""));
		panel.setMinimumSize(new Dimension((width/4)*3,height));
		
		// button for clearing show (turn off all lights)
		clearDrawTestButton = new JButton("Clear");
		clearDrawTestButton.addActionListener(this);
		clearDrawTestButton.setMaximumSize(new Dimension(120, 20));
		panel.add(clearDrawTestButton, "wrap");
		
		return panel;
	}
	
	public JComponent makeTraceTestPanel(){
		JPanel panel = new JPanel(false);
        panel.setLayout(new MigLayout(""));
		panel.setMinimumSize(new Dimension((width/4)*3,height));
		
		// slider to adjust speed of tracer
		panel.add(new Label("Tracer Speed", Label.RIGHT));
		traceSpeedSlider = new Scrollbar(Scrollbar.HORIZONTAL, 0, 1, 0, 100);
		traceSpeedSlider.setForeground(Color.black);
		traceSpeedSlider.setBackground(Color.white);
		traceSpeedSlider.setMinimumSize(new Dimension(200, 16));
		panel.add(traceSpeedSlider, "wrap");
		
		return panel;
	}
	
	public JComponent makeWaveShowPanel(){
		JPanel panel = new JPanel(false);
        panel.setLayout(new MigLayout(""));
		panel.setMinimumSize(new Dimension((width/4)*3,height));
		
		waveListModel = new DefaultListModel();
		waveList = new JList(waveListModel);
		waveList.setMinimumSize(new Dimension(100,height));
		waveList.addListSelectionListener(this);
		panel.add(waveList, "west");
		
		// sub-panel to hold sliders
		JPanel sliderpanel = new JPanel(false);
		sliderpanel.setLayout(new MigLayout(""));

		
		// slider for adjusting damping value
		sliderpanel.add(new Label("Damping", Label.RIGHT));
		dampingSlider = new Scrollbar(Scrollbar.HORIZONTAL, 0, 1, 0, 100);
		dampingSlider.setForeground(Color.black);
		dampingSlider.setBackground(Color.white);
		dampingSlider.setMinimumSize(new Dimension(200, 16));
		sliderpanel.add(dampingSlider, "wrap");
		
		// slider for adjusting nonlinearity value
		sliderpanel.add(new Label("Nonlinearity", Label.RIGHT));
		fpuSlider = new Scrollbar(Scrollbar.HORIZONTAL, 0, 1, 0, 100);
		fpuSlider.setForeground(Color.black);
		fpuSlider.setBackground(Color.white);
		fpuSlider.setMinimumSize(new Dimension(200, 16));
		sliderpanel.add(fpuSlider, "wrap");
		
		// slider for adjusting y offset of wave surface
		sliderpanel.add(new Label("Y-Offset", Label.RIGHT));
		yoffsetSlider = new Scrollbar(Scrollbar.HORIZONTAL, 60, 1, 0, 100);
		yoffsetSlider.setForeground(Color.black);
		yoffsetSlider.setBackground(Color.white);
		yoffsetSlider.setMinimumSize(new Dimension(200, 16));
		sliderpanel.add(yoffsetSlider, "wrap");
		
		// slider for adjusting mysterious dx value (relates to horizontal wave speed)
		sliderpanel.add(new Label("DX value", Label.RIGHT));
		dxSlider = new Scrollbar(Scrollbar.HORIZONTAL, 2, 1, 0, 100);
		dxSlider.setForeground(Color.black);
		dxSlider.setBackground(Color.white);
		dxSlider.setMinimumSize(new Dimension(200, 16));
		sliderpanel.add(dxSlider, "wrap");
		
		panel.add(sliderpanel, "west");
		
		JPanel buttonPanel = new JPanel(false);
		buttonPanel.setLayout(new MigLayout(""));
		

		// slider for adjusting mysterious c value (relates to wave speed)
		buttonPanel.add(new Label("C value", Label.RIGHT));
		cSlider = new Scrollbar(Scrollbar.HORIZONTAL, 12, 1, 0, 100);
		cSlider.setForeground(Color.black);
		cSlider.setBackground(Color.white);
		cSlider.setMinimumSize(new Dimension(200, 16));
		buttonPanel.add(cSlider, "wrap, span 2");
		
		// slider for adjusting brightness
		buttonPanel.add(new Label("Brightness", Label.RIGHT));
		brightnessSlider = new Scrollbar(Scrollbar.HORIZONTAL, 255, 1, 0, 255);
		brightnessSlider.setForeground(Color.black);
		brightnessSlider.setBackground(Color.white);
		brightnessSlider.setMinimumSize(new Dimension(200, 16));
		buttonPanel.add(brightnessSlider, "wrap, span 2");
		
		// slider for adjusting alpha
		buttonPanel.add(new Label("Alpha", Label.RIGHT));
		alphaSlider = new Scrollbar(Scrollbar.HORIZONTAL, 255, 1, 0, 255);
		alphaSlider.setForeground(Color.black);
		alphaSlider.setBackground(Color.white);
		alphaSlider.setMinimumSize(new Dimension(200, 16));
		buttonPanel.add(alphaSlider, "wrap, span 2");
		
		buttonPanel.add(new Label("", Label.RIGHT));
		// button for resetting wave after it goes haywire
		resetWaveButton = new JButton("Reset Wave");
		resetWaveButton.addActionListener(this);
		resetWaveButton.setMaximumSize(new Dimension(120, 20));
		buttonPanel.add(resetWaveButton);
		
		// button for saving sprite properties for future loading
		saveWavesButton = new JButton("Save Waves");
		saveWavesButton.addActionListener(this);
		saveWavesButton.setMaximumSize(new Dimension(120, 20));
		buttonPanel.add(saveWavesButton, "wrap");
		
		panel.add(buttonPanel, "west");
		
		return panel;
	}
	
	public JComponent makeDisplayModePanel(){
		JPanel panel = new JPanel(false);
        panel.setLayout(new MigLayout("insets 0 5 0 0"));
		panel.setMinimumSize(new Dimension(width/4,height));
		
		// drop down list to select raster display mode
		panel.add(new Label("Display Mode:"), "wrap");
		JComboBox displayModeList = new JComboBox(new String[] {"Raster","Raster + Detectors", "Detector Values"});
		displayModeList.setSelectedIndex(1);
		displayModeList.addActionListener(this);		
		panel.add(displayModeList);
		
		// check box to tint raster display blue
		tintBlueButton = new JCheckBox("Tint Blue");
		tintBlueButton.setSelected(true);
		tintBlueButton.addItemListener(this);
		panel.add(tintBlueButton, "wrap");
		
		return panel;
	}
	
	public void saveWaves(){
		Completable a  = main.getCurrentAnimation();
		if(a instanceof WaveShow){			// confirm show is WaveShow
			try{ 
			    FileWriter fstream = new FileWriter("depends//waves.properties");	// create file
			    BufferedWriter out = new BufferedWriter(fstream);
			    out.write("<?xml version=\"1.0\"?>\n");
			    out.write("<waves>\n");
		   
				ConcurrentHashMap<Integer, Wave> waves = ((WaveShow) a).getWaves();
				Iterator<Wave> iter = waves.values().iterator();
				while(iter.hasNext()){
					Wave wave = iter.next();
					out.write("<wave>\n");
					out.write("\t<damping>");
					out.write(String.valueOf(wave.getDamping()));
					out.write("</damping>\n");
					out.write("\t<nonlinearity>");
					out.write(String.valueOf(wave.getNonlinearity()));
					out.write("</nonlinearity>\n");
					out.write("\t<yoffset>");
					out.write(String.valueOf(wave.getYoffset()));
					out.write("</yoffset>\n");
					out.write("\t<dx>");
					out.write(String.valueOf(wave.getDX()));
					out.write("</dx>\n");
					out.write("\t<c>");
					out.write(String.valueOf(wave.getC()));
					out.write("</c>\n");
					out.write("\t<brightness>");
					out.write(String.valueOf(wave.getBrightness()));
					out.write("</brightness>\n");
					out.write("\t<alpha>");
					out.write(String.valueOf(wave.getAlpha()));
					out.write("</alpha>\n");
					out.write("\t<points>");
					double[][] points = wave.getPoints();
					for(int i=0; i<points.length; i++){
						for(int n=0; n<points[i].length; n++){
							out.write(String.valueOf(points[i][n]));
							if(n < points[i].length-1){
								out.write(":");
							}
						}
						if(i < points.length - 1){
							out.write(",");
						}
					}
					out.write("</points>\n");
					out.write("</wave>\n");
				}
				out.write("</waves>\n");
			    out.close();	// close the output stream
			 } catch (Exception e){
			   	System.err.println("Error: " + e.getMessage());
			 }
		}
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
		} else if(e.getActionCommand().equals("Save Waves")){
			saveWaves();
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
		if(e.target instanceof Scrollbar){
			if(currentWaveID != -1){					// if a wave sprite has been selected in the wave list...
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
				int counter = 1;
				while(iter.hasNext()){
					Wave wave = iter.next();
					waveListModel.addElement("Wave "+wave.getID());
					counter++;
				}
			}
		}
	}

	public void valueChanged(ListSelectionEvent e) {
		if (e.getValueIsAdjusting() == false) {		// done adjusting
	        if(waveList.getSelectedIndex() >= 0){		// if mouse event
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

	public void itemStateChanged(ItemEvent e) {
		if(e.getItemSelectable() == tintBlueButton){
			if(e.getStateChange() == ItemEvent.DESELECTED){
				main.rasterPanel.enableTint(false);
			} else {
				main.rasterPanel.enableTint(true);
			}
		}
	}
	
	public int getCurrentWaveID(){
		return currentWaveID;
	}
	
}
