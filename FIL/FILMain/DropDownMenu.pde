public class DropDownMenu extends Widget implements WidgetListener{
  
  // DROPDOWNMENU.pde
  // encapsulates many buttons and controls when they're displayed/accessable.
  // also manages the receipt of widget events from buttons and broadcasts them
  // back to the widget manager.
  
  private int buttonWidth, buttonHeight, leading;
  private ArrayList items;
  private boolean displayItems = false;
  private boolean on = false;
  private PImage img, imgDown;
  
  public DropDownMenu(String name, int x, int y, float value, int buttonHeight, int leading, PImage img, PImage imgDown){
    super(name, x, y, value);
    this.w = img.width;
    this.h = img.height;
    this.buttonWidth = w;
    this.buttonHeight = buttonHeight;
    this.leading = leading;
    this.img = img;
    this.imgDown = imgDown;
    this.items = new ArrayList();
  }
  
  public void addItem(String name, float val){
    Button b = new Button(name, x, y + (buttonHeight+leading)*items.size(), buttonWidth, buttonHeight, val);
    b.addListener(this);
    items.add(b);
    //super.h = ((buttonHeight+leading)*items.size())-leading;
  }
  
  public void draw(){
    pushMatrix();
    translate(x, y);
    if(on){
      image(imgDown,0,0);
    } else {
      image(img,0,0);
    }
    if(displayItems){
      Iterator i = items.iterator();
      while(i.hasNext()){
        Button b = (Button)i.next();
        b.draw();
      }
    }
    popMatrix();
  }
  
  public void dragged(){
    // probably need to inform drop down items from here
  }
  
  public void pressed(){
    if(displayItems){
      Iterator i = items.iterator();
      while(i.hasNext()){
        ((Button)i.next()).mousePressed(mouseX-img.height, mouseY-img.height);
      }
      displayItems = false;
      h = img.height;
    } else {
      displayItems = true;
      h = img.height + ((buttonHeight+leading)*items.size())-leading;
    }
  }
  
  public void released(){
  }
  
  public void rollOver(){
    // highlight
  }
  
  public void rollOut(){
    displayItems = false;
  }
  
  public void silentOff(){
    on = false;
  }
  
  public void cursorMovement(){
    // must report mouse at all times, not just when cursor enters/exits radio button area
    Iterator i = items.iterator();
    while(i.hasNext()){
      //((Button)i.next()).mouseMoved(mouseX-x, mouseY-y);
      ((Button)i.next()).mouseMoved(mouseX-img.height, mouseY-img.height);
    }
  }
  
  public void widgetEvent(WidgetEvent we){
    value = we.widget.value;
    ((Button)we.widget).silentOff();  // prevent button from staying highlighted
    WidgetEvent newwe = new WidgetEvent(this, PRESSED, true);
    super.newEvent(newwe);
    on = true;
  }
  
}
