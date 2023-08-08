---
layout: page
title: react-dynamic-renderer
description: ReactJS module that renders flexible JSON schemas into fully interactive React components.
importance: 2
category: software
github: https://github.com/TheMatrixMaster/react-dynamic-renderer
---

### Usage

Add a call to `setupRenderer` in your project's `index.js`:

```typescript
import { setupRenderer } from "@heroai-team/api/dist/src/rendering";

// import the react components that you want to
// dynamically render from the schema
import { MyComponent, ... } from 'path.to.components';

setupRenderer({
  myComponent: {
    element: MyComponent,
    transform: ({
      component,
      parseDynAction,
      parseDynVariable,
      parseDynStyle,
      renderer
    }) => {
      // compute derived props
      const parsedProps = {
        // parseDynVariable will resolve a DynamicVariable type
        label: parseDynVariable(component.props.label),
        // parseDynAction will resolve a DynamicAction type
        onPress: () => parseDynAction(component.props.onPress),
        // renderer is an instance of main renderer. You can use it to render child components
        subtitle: () => renderer({ component: component.props.subtitle }),
      }

      return { parsedProps, useStyle: boolean };
    }
  },
  ...
})
```

To use the renderer in your project, you need to call the `useRenderer`
hook inside your React component and provide the desired renderer config
parameters:

```typescript
import React from 'react';
import { SERVER } from '@env';
import { useRenderer } from "@heroai-team/api/dist/src/rendering";

const MyReactFunction = (props) => {
  // retrieve the styling variables that I want to use
  const colors = {
    white: '#fff',
    black: '#000',
    ...
  }
  const fontFamilies = {
    comfortaa: 'font.asset',
    lato: 'font.asset',
    ...
  }
  const dimensions = {
    screenHeight: 300,
    screenWidth: 100,
  }

  // retrieve the variables that I want to use
  const [state, setState] = React.useState({});
  const variables = props.beacon.variables;

  // declare the useRenderer hook
  const renderer = useRenderer({
    styleMap: {
      colors,
      dimensions,
      fontFamilies,
    },
    variableMap: {
      state,
      props,
      variables,
      env: { serverURL: SERVER },
    },
    actionMap: {
      setState: ({ action }) => setState({ ...state, [action.payload.key]: value }),
      popNavigation: () => props.navigation.pop(),
      handleURL: ({ action, parseDynVariable }) => {
        const url = parseDynVariable(action.payload.link);
        return openURL(url);
      }
    },
  })

  // let's render MyComponent defined above
  const TextComponent = {
    type: "myComponent",
    style: {
      "marginTop": 5,
    },
    props: {
      label: "Hello world!",
      onPress: {
        name: "popNavigation",
      }
    }
  }

  return renderer(TextComponent);
}
```
