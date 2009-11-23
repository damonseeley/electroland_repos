public class WidgetManager implements WidgetListener {
  
  // WIDGETMANAGER.pde
  // gets called by PApplet core to communicate cursor interaction.
  // also receives widget events from widgets in order to determine the
  // next course of action.
  
  private TCPClient client;   // reference to mpe broadcaster
  private ArrayList widgets;  // root level widgets
  private boolean enableSounds;
 
  public WidgetManager(TCPClient client, Sample interfaceSoundSample, boolean enableSounds){
    this.client = client;
    this.enableSounds = enableSounds;
    widgets = new ArrayList();
  }
  
  public void addItem(Widget w){
    w.addListener(this);
    widgets.add(w);
  }
  
  public void removeItem(Widget w){
    if(widgets.contains(w)){
      w.removeListener(this);
      widgets.remove(w);
    }
  }
  
  public void draw(){
    Iterator i = widgets.iterator();
    while (i.hasNext()){
      ((Widget)i.next()).draw();
    }
  }
  
  public boolean isOverAWidget(int mouseX, int mouseY){
    Iterator i = widgets.iterator();
    while (i.hasNext()){
      if(((Widget)i.next()).mouseInside(mouseX, mouseY)){
        return true;
      }
    }
    return false;
  }
  
  public void pressed(int mouseX, int mouseY){
    Iterator i = widgets.iterator();
    while (i.hasNext()){
      ((Widget)i.next()).mousePressed(mouseX, mouseY);
    }
  }
  
  public void released(int mouseX, int mouseY){
    Iterator i = widgets.iterator();
    while (i.hasNext()){
      ((Widget)i.next()).mouseReleased(mouseX, mouseY);
    }
  }
  
  public void dragged(int mouseX, int mouseY){
    Iterator i = widgets.iterator();
    while (i.hasNext()){
      Widget w = (Widget)i.next();
      w.mouseDragged(mouseX, mouseY);
      w.mouseMoved(mouseX, mouseY);
    }
  }
  
  public void cursorMovement(int mouseX, int mouseY){
    Iterator i = widgets.iterator();
    while (i.hasNext()){
      ((Widget)i.next()).mouseMoved(mouseX, mouseY);
    }
  }
 
  public void widgetEvent(WidgetEvent we){
    //println(we.name);
    if(we.name.equals("English")){
      if(!interfaceSoundSample.isPlaying() && enableSounds){
        interfaceSoundSample.play();
      }
      client.broadcast("buttonEvent,english");
    } else if(we.name.equals("Espanol")){
      if(!interfaceSoundSample.isPlaying() && enableSounds){
        interfaceSoundSample.play();
      }
      client.broadcast("buttonEvent,espanol");
    } else if(we.name.equals("Author Cloud")){
      client.broadcast("buttonEvent,cloud,normal");
    } else if(we.name.equals("By Date")){
      client.broadcast("buttonEvent,cloud,date");
    } else if(we.name.equals("By Popularity")){
      client.broadcast("buttonEvent,cloud,popularity");
    } else if(we.name.equals("By Genre")){
      client.broadcast("buttonEvent,cloud,genre,"+int(we.widget.value));
    } else if(we.name.equals("ZoomIn")){
      if(!interfaceSoundSample.isPlaying() && enableSounds){
        interfaceSoundSample.play();
      }
      client.broadcast("buttonEvent,zoomin");
    } else if(we.name.equals("ZoomOut")){
      if(!interfaceSoundSample.isPlaying() && enableSounds){
        interfaceSoundSample.play();
      }
      client.broadcast("buttonEvent,zoomout");
    } else if(we.name.equals("Balloon")){
      if(!interfaceSoundSample.isPlaying() && enableSounds){
        interfaceSoundSample.play();
      }
      if(((Balloon)we.widget).quoteMode){
        client.broadcast("quote,fadein,"+((Balloon)we.widget).author.getID());
      } else {
        client.broadcast("bio,fadein,"+((Balloon)we.widget).author.getID());
      }      
    } else if(we.name.equals("Slider")){
      client.broadcast("buttonEvent,slide,"+((Slider)we.widget).getBarPosition());
    }
  } 
  
}
