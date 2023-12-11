var isOnPlatform = function(obj) {{
    return obj.y < 325;
  }}
  
  var isNotOnPlatform = function(obj) {{
    return !(obj.y < 325);
  }}
  
  var isLeftOfPlatform = function(obj) {{
    return obj.y > 325 && obj.x < 175;
  }}
  
  var isNotLeftOfPlatform = function(obj) {{
    return !(obj.y > 325 && obj.x < 175);
  }}
  
  var isRightOfPlatform = function(obj) {{
    return obj.y > 325 && obj.x > 425;
  }}
  
  var isNotRightOfPlatform = function(obj) {{
    return !(obj.y > 325 && obj.x > 425);
  }}
  
  var isPurple = function(obj) {{
    return obj.color=="purple";
  }}
  
  var isRed = function(obj) {{
    return obj.color=="red";
  }}
  
  var isLarge = function(obj) {{
    return obj.dims[0] > 25 || obj.dims[1] > 25;
  }}
  
  var isNotLarge = function(obj) {{
    return !(obj.dims[0] > 25 || obj.dims[1] > 25);
  }}