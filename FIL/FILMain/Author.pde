public class Author extends TextBlock {
  
  // AUTHOR.pde
  // the visual representation as well as metadata about the author.
  
  private String quote = "";
  public boolean triggered = false;
  private boolean rolledOver = false;
  private float targetTextScale, targetOpacity, defaultAlphaVal;
  private float selectedRedVal;
  private float selectedGreenVal;
  private float selectedBlueVal;
  //private int fadeInDuration;
  //private int holdDuration;
  //private int fadeOutDuration;
  private WidgetManager widgetManager;
  private Balloon balloon;  // reference to controls
  private float xtarget, ytarget;  // used for damped to target movement, such as genre/date/popularity mode
  private float xsource, ysource;
  private boolean dampedMovement = false;
  private int movementCounter = 0;
  private int movementDuration;
  private float maxOverlap;  // used to drop alpha when overlapping quotes
  private float textScaleTarget;
  private int textScaleDuration;
  private int textScaleCounter = 0;
  private boolean tweenScale = false;
  private boolean tweenDamping = false;
  private float targetXdamping, targetYdamping;
  private float originalXdamping, originalYdamping;
  
  public String seminalwork_english, seminalwork_spanish;
  public String bio_english, bio_spanish;
  public String quote_english, quote_spanish;
  public String[] genres;
  public String[] geographies;
  public int born, died, workbegan, workended, popularity;
  
  public Author(int id, String name, float x, float y, String fontName, int fontSize, float textScale){
    super(id, name, x, y, fontName, fontSize, textScale);
    originalXdamping = xdamping;
    originalYdamping = ydamping;
    targetXdamping = 0.8;
    targetYdamping = 0.8;
  }
  
  public Author(int id, String name, String quote, float x, float y, String fontName, int fontSize, float textScale){
    super(id, name, x, y, fontName, fontSize, textScale);
    this.quote = quote;
    originalXdamping = xdamping;
    originalYdamping = ydamping;
    targetXdamping = 0.8;
    targetYdamping = 0.8;
  }
  
  public void applyGravity(float xGrav, float yGrav, float xgravity, float ygravity){
    if(!triggered){
      super.applyGravity(xGrav, yGrav, xgravity, ygravity);
    }
  }
  
  public void addControls(WidgetManager widgetManager, Balloon balloon){
    this.widgetManager = widgetManager;
    this.balloon = balloon;
  }
  
  public void checkCollisions(ConcurrentHashMap textBlocks, float xMargin, float yMargin){
    // only be able to slide horizontally.
    // only push against other text blocks, never get pushed.
    this.xMargin = xMargin;
    this.yMargin = yMargin;
    maxOverlap = 1;  // start at 1, go as low as 0
    
    Iterator iter = textBlocks.values().iterator();
    while(iter.hasNext()){                      // loop through all textBlocks
      TextBlock b = (TextBlock)iter.next();
      if(b.getID() != id){                      // if older than this block, check if there is an overlap
        //if((abs(x - b.getX()) < abs(w*0.5 + b.getWidth()*0.5)) && (abs(y - b.getY()) < abs(h*0.5 + b.getHeight()*0.5))){
        if((abs(x - b.getX()) < abs((w*0.5 + b.getWidth()*0.5) + (xMargin*2))) && (abs(y - b.getY()) < abs((h*0.5 + b.getHeight()*0.5) + (xMargin*2)))){
          
          //float xoverlap = abs(w*0.5 + b.getWidth()*0.5) - abs(x - b.getX());    // no margins
          //float yoverlap = abs(h*0.5 + b.getHeight()*0.5) - abs(y - b.getY());
          float xoverlap = abs((w*0.5)+xMargin + (b.getWidth()*0.5)+xMargin) - abs(x - b.getX());
          float yoverlap = abs((h*0.5)+yMargin + (b.getHeight()*0.5)+yMargin) - abs(y - b.getY());
          
          if(b instanceof QuoteLine){
            // compare area of overlap to total area to get percentage alpha should be
            float overlapPercentage = (xoverlap * yoverlap) / (w * h);
            maxOverlap -= overlapPercentage;
            if(maxOverlap < 0){
              maxOverlap = 0;
            }
          }
          
          /*
          // check if opposing textblock is a quote, and change alpha value based on amount of overlap
          if(b instanceof QuoteLine){
            //float xdist = abs(((QuoteLine)b).getParagraphCenterX() - x) / (((QuoteLine)b).getParagraphWidth()/2);
            //float ydist = abs(((QuoteLine)b).getParagraphCenterY() - y) / (((QuoteLine)b).getParagraphHeight()/2);
            float xdist = abs(((QuoteLine)b).getParagraphCenterX() - x) / (((QuoteLine)b).getParagraphWidth()/2);
            float ydist = abs(((QuoteLine)b).getParagraphCenterY() - y) / (((QuoteLine)b).getParagraphHeight()/2);
            float paragraphHypo = sqrt(sq(((QuoteLine)b).getParagraphWidth()) + sq(((QuoteLine)b).getParagraphHeight()));
            //maxOverlap = sqrt((xdist*xdist) + (ydist*ydist));
            //println("xdist: " + xdist + " ydist: "+ ydist);
            if(maxOverlap > 1){
              maxOverlap = 1;
            }
            println(maxOverlap);
          }
          */
          
          /*
          float opposingAlpha = b.getAlpha() / 255;
          //float totalOverlap = ((w - xoverlap) / w) * ((h - yoverlap) / h);
          float totalOverlap = ((w - xoverlap) / w);
          if(b instanceof QuoteLine){
            if(totalOverlap * opposingAlpha < maxOverlap){
              maxOverlap = totalOverlap * opposingAlpha;
              println(maxOverlap);
            }
          }
          */
          
          float thisXvel = xv;  // make copies as they'll be modified simultaneously
          float thisYvel = yv;  
          float otherXvel = b.getXVel();
          float otherYvel = b.getYVel();
          
          if(y > b.getY()){    // this is below other textblock
            if(xoverlap > yoverlap){
              if(!triggered){
                yv += yoverlap * 0.5 * verticalSpring * b.getScale();
              }
              b.push(0, 0 - yoverlap * 0.5 * verticalSpring * textScale);
            } else {
              if(!triggered){
                yv += xoverlap * 0.5 * verticalSpring * b.getScale();
              }
              b.push(0, 0 - xoverlap * 0.5 * verticalSpring * textScale);
            }
          } else {             // other textblock is below this
            if(xoverlap > yoverlap){
              if(!triggered){
                yv -= yoverlap * 0.5 * verticalSpring * b.getScale();
              }
              b.push(0, yoverlap * 0.5 * verticalSpring * textScale);
            } else {
              if(!triggered){
                yv -= xoverlap * 0.5 * verticalSpring * b.getScale();
              }
              b.push(0, xoverlap * 0.5 * verticalSpring * textScale);
            }
          }
          
          if(x > b.getX()){    // this is to the right of the other textblock
            if(xoverlap > yoverlap){
              if(!triggered){
                xv += yoverlap * 0.5 * horizontalSpring * b.getScale();
              }
              b.push(0 - yoverlap * 0.5 * horizontalSpring * textScale, 0);
            } else {
              if(!triggered){
                xv += xoverlap * 0.5 * horizontalSpring * b.getScale();
              }
              b.push(0 - xoverlap * 0.5 * horizontalSpring * textScale, 0);
            }
          } else {             // textblock is to the right of this
            if(xoverlap > yoverlap){
              if(!triggered){
                xv -= yoverlap * 0.5 * horizontalSpring * b.getScale();
              }
              b.push(yoverlap * 0.5 * horizontalSpring * textScale, 0);
            } else {
              if(!triggered){
                xv -= xoverlap * 0.5 * horizontalSpring * b.getScale();
              }
              b.push(xoverlap * 0.5 * horizontalSpring * textScale, 0);
            }
          }
          
        }
      }
    }
  }
  
  public void move(int stageWidth, int overflow){
    if(dampedMovement){
      if(movementCounter < movementDuration){
        float progress = sin((movementCounter / (float)movementDuration) * (PI/2));
        x = xsource + ((xtarget-xsource) * progress);
        y = ysource + ((ytarget-ysource) * progress);
        movementCounter++;
      } else {
        dampedMovement = false;
        movementCounter = 0;
        tweenDamping = true;
        xdamping = targetXdamping;
        ydamping = targetYdamping;
      }
    } else {
      if(tweenDamping){
        if(movementCounter < movementDuration){
          float progress = (movementCounter / (float)movementDuration);
          xdamping = targetXdamping - ((targetXdamping - originalXdamping) * progress);
          ydamping = targetYdamping - ((targetYdamping - originalYdamping) * progress);
          movementCounter++;
        } else {
          xdamping = originalXdamping;
          ydamping = originalYdamping;
          movementCounter = 0;
          tweenDamping = false;
        }
      }
      x += xv;
      if(!triggered){
        y += yv;
      }
      xv *= xdamping;
      yv *= ydamping;
      if(x-(w*0.5) > stageWidth + overflow){    // wrap text block to other side of the screen
        x = (0-overflow)+10;
      } else if(x+(w*0.5) < -overflow) {
        x = stageWidth+overflow-10;
      }
    }
  }
  
  public void moveTo(float xtarget, float ytarget, int movementDuration){
    this.xtarget = xtarget;
    this.ytarget = ytarget;
    xsource = x;
    ysource = y;
    this.movementDuration = movementDuration;
    dampedMovement = true;
  }
  
  public void scaleTo(float textScaleTarget, int textScaleDuration){
    println(textScaleTarget +" "+textScaleDuration);
    this.textScaleTarget = textScaleTarget;
    this.textScaleDuration = textScaleDuration;
    textScaleCounter = 0;
    tweenScale = true;
  }
  
  public boolean hasQuote(String language){
    if(language == "English"){
      if(quote_english.length() > 0){
        return true;
      }
    } else {
      if(quote_spanish.length() > 0){
        return true;
      }
    }
    return false;
  }
  
  public boolean hasBio(String language){
    if(language == "English"){
      if(bio_english.length() > 0){
        return true;
      }
    } else {
      if(bio_spanish.length() > 0){
        return true;
      }
    }
    return false;
  }
  
  public String getQuote(){
    return quote;
  }
  
  public void retrigger(){
    fadeOut = false;
    fadeIn = false;
    hold = true;
    counter = 0;
  }
  
  public void deactivate(){
    if(!fadeOut){
      fadeOut = true;
      fadeIn = false;
      hold = false;
      counter = 0;
    }
  }
  
  public void release(){
    //if(pressed && !triggered && hasQuote(userLanguage)){
    if(pressed && !triggered){
      triggered = true;
      fadeIn = true;
      counter = 0;
    }
    if(rolledOver || scaleDown){
      println(textValue + " released");
      rolledOver = false;
      scaleDown = false;
      scaleUp = true;
      counter = 0;
    }
    super.release();
  }
  
  public void releasedOutside(){
    if(rolledOver){
      println(textValue + " released outside");
      rolledOver = false;
      scaleDown = false;
      scaleUp = true;
      counter = 0;
    }
    super.releasedOutside();
  }
  
  public void rollOver(){
    if(!rolledOver){
      println(textValue + " rolled over");
      scaleDown = true;
      rolledOver = true;
      targetTextScale = defaultTextScale * rollOverScale;
      targetOpacity = alpha(c) * rollOverAlpha;
      counter = 0;
    }
  }
  
  public void rollOut(){
    if(rolledOver){
      println(textValue + " rolled out");
      scaleDown = false;
      rolledOver = false;
      scaleUp = true;
      counter = 0;
    }
  }
  
  public void render(PGraphicsOpenGL pgl, int xoffset, int yoffset, boolean yflip){
    int hh = stageHeight/2;
    if(y > hh){
      defaultAlphaVal = alphaStartVal - ((y-hh)/hh)*alphaFallOffVal;
    } else {
      defaultAlphaVal = alphaStartVal - ((hh-y)/hh)*alphaFallOffVal;
    }
    
    if(tweenScale){
      textScaleCounter++;
      float progress = sin((textScaleCounter / (float)textScaleDuration) * (PI/2));
      if(textScaleCounter > textScaleDuration){
        defaultTextScale = textScale = textScaleTarget;
        println(textScale);
        tweenScale = false;
        textScaleCounter = 0;
      }
      textScale = defaultTextScale + ((textScaleTarget - defaultTextScale) * progress);
      w = textRender.getWidth(textValue) * textScale;
      h = textRender.getHeight(textValue) * textScale;
    }
    
    // apply overlap value to transparency
    defaultAlphaVal *= maxOverlap;
    //println("overlap %: "+maxOverlap +" default alpha: "+defaultAlphaVal);
    
    if(scaleDown){                                     // ROLLED OVER
      float progress = sin((counter / (float)rollOverDuration) * (PI/2));
      textScale = defaultTextScale + (targetTextScale - defaultTextScale) * progress;
      alphaVal = defaultAlphaVal - (defaultAlphaVal - targetOpacity) * progress;
      if(counter == rollOverDuration){
        scaleDown = false;
        textScale = targetTextScale;
        alphaVal = targetOpacity;
      }
      w = textRender.getWidth(textValue) * textScale;
      h = textRender.getHeight(textValue) * textScale;
    } else if(scaleUp && textScale < defaultTextScale){       // ROLLED OUT
      float progress = sin((counter / (float)rollOutDuration) * (PI/2));
      textScale = targetTextScale + (defaultTextScale - targetTextScale) * progress;
      alphaVal = targetOpacity + (defaultAlphaVal - targetOpacity) * progress;
      if(counter == rollOutDuration){
        scaleUp = false;
        textScale = defaultTextScale;
        alphaVal = defaultAlphaVal;
      }
      w = textRender.getWidth(textValue) * textScale;
      h = textRender.getHeight(textValue) * textScale;
    } else if(rolledOver){                                     // HOLDING OVER
      // holding while cursor over
    } else {                                                   // REGULAR MODE
      alphaVal = defaultAlphaVal;
    }
    
    /*
    // THIS DOES HORIZONTAL ALPHA FADING
    if(x-(w/2) < horizontalFallOff){
      alphaVal = defaultAlphaVal * ((float)(x-(w/2))/horizontalFallOff);
    } else if(x+(w/2) > stageWidth-horizontalFallOff){
      alphaVal = defaultAlphaVal * ((float)(stageWidth - (x+(w/2)))/horizontalFallOff);
    } else if(x-(w/2) < 0 || x+(w/2) > stageWidth){
      alphaVal = 0;
    } else {
      alphaVal = defaultAlphaVal;
    }
    */
    
    if(triggered){
      if(fadeIn){
        float progress = sin((counter / (float)fadeInDuration) * (PI/2));
        if(counter == fadeInDuration){
          fadeIn = false;
          hold = true;
          counter = 0;
        }
        
        if(defaultTextScale < maxTextScale){  // if less than max text scale...
          textScale = defaultTextScale + ((maxTextScale - defaultTextScale) * progress);
          w = textRender.getWidth(this.textValue) * textScale;    // necessary to keep it centered
          h = textRender.getHeight(this.textValue) * textScale;
        }
        
        c = color((selectedRedVal - redVal)*progress + redVal, (selectedGreenVal - greenVal)*progress + greenVal, (selectedBlueVal - blueVal)*progress + blueVal, (255 - alphaVal)*progress+ alphaVal);
        //c = color((selectedRedVal - redVal)*progress + redVal, (selectedGreenVal - greenVal)*progress + greenVal, (selectedBlueVal - blueVal)*progress + blueVal, alphaVal);
        super.render(pgl, xoffset, yoffset, yflip);
      } else if(hold){
        float progress = counter / (float)holdDuration;
        if(counter == holdDuration){
          hold = false;
          fadeOut = true;
          counter = 0;
          if(balloon != null){
            balloon.fadeOut(fadeOutDuration);
          }
        }
        c = color(selectedRedVal, selectedGreenVal, selectedBlueVal, 255);
        //c = color(selectedRedVal, selectedGreenVal, selectedBlueVal, alphaVal);
        super.render(pgl, xoffset, yoffset, yflip);
      } else if(fadeOut){
        float progress = 1 - sin((counter / (float)fadeOutDuration) * (PI/2));
        
        if(defaultTextScale < maxTextScale){  // if less than max text scale...
          textScale = defaultTextScale + ((maxTextScale - defaultTextScale) * progress);
          w = textRender.getWidth(textValue) * textScale;
          h = textRender.getHeight(textValue) * textScale;
        }
        
        if(counter == fadeOutDuration){
          fadeOut = false;
          triggered = false;
          counter = 0;
          if(widgetManager != null && balloon != null){
            widgetManager.removeItem(balloon);
          }
        }
        
        c = color((selectedRedVal - redVal)*progress + redVal, (selectedGreenVal - greenVal)*progress + greenVal, (selectedBlueVal - blueVal)*progress + blueVal, (255 - alphaVal)*progress+ alphaVal);
        //c = color((selectedRedVal - redVal)*progress + redVal, (selectedGreenVal - greenVal)*progress + greenVal, (selectedBlueVal - blueVal)*progress + blueVal, alphaVal);
        super.render(pgl, xoffset, yoffset, yflip);
      }
    } else {
      c = color(redVal, greenVal, blueVal, alphaVal);
      super.render(pgl, xoffset, yoffset, yflip);
    }
    counter++;
  }
  
  public void softenCollisions(){
    movementCounter = 0;
    tweenDamping = true;
    xdamping = targetXdamping;
    ydamping = targetYdamping;
  }
  
  public void setFadeInDuration(int fadeInDuration){
    this.fadeInDuration = fadeInDuration;
  }
  
  public void setHoldDuration(int holdDuration){
    this.holdDuration = holdDuration;
  }
  
  public void setFadeOutDuration(int fadeOutDuration){
    this.fadeOutDuration = fadeOutDuration;
  }
  
  public void setSelectedRed(float selectedRedVal){
    this.selectedRedVal = selectedRedVal;
  }
  
  public void setSelectedGreen(float selectedGreenVal){
    this.selectedGreenVal = selectedGreenVal;
  }
  
  public void setSelectedBlue(float selectedBlueVal){
    this.selectedBlueVal = selectedBlueVal;
  }
  
  public void push(float xforce, float yforce){
    if(!triggered){
      yv += yforce;
      xv += xforce;
    }
  }
  
}
