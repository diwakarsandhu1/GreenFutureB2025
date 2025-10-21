
import { v4 as uuidv4 } from "uuid";

export const OptionsData = [
    {
      id: "1",
      name: "Minimizing investment in fossil fuel producers",
    },
    {
      id: "2",
      name: "Minimizing investment in weapons manufacut eres",
    },
    {
      id: "3",
      name: "Investing in environmentally friendly companies (Learn more)",
    },
    {
      id: "4",
      name: "Investing in companies with positive social impacts (Learn more)",
    },
    {
      id: "5",
      name: "Investing in companies with string goernance (Learn more)",
    },
  ];
  
  export const columnsFromBackend = {
    [uuidv4()]: {
      title: 'Most Important',
      items: OptionsData,
    },
    [uuidv4()]: {
      title: 'Moderate Important',
      items: [],
    },
    [uuidv4()]: {
      title: 'Least Important',
      items: [],
    },
  };