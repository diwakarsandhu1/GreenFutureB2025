import * as React from "react";
import {
  Box,
  Card,
  CardHeader,
  Grid,
  Chip,
  Badge,
  IconButton,
  Typography,
  CardContent,
  CardActions,
  Avatar,
} from "@mui/material";
import { Draggable } from "react-beautiful-dnd";
import MoreVertIcon from "@mui/icons-material/MoreVert";
import NotificationsNoneIcon from "@mui/icons-material/NotificationsNone";
import { styled } from "@mui/system";




const OptionCard = ({ item, index }) => {
    return (
      <Draggable key={item.id} draggableId={item.id} index={index}>
        {(provided) => (
          <div
            ref={provided.innerRef}
            {...provided.draggableProps}
            {...provided.dragHandleProps}
          >
            <Card sx={{ minWidth: 275, m: "8px 1px" }}>
              <CardContent>
                <Typography variant="h6">{item.name}</Typography>
              </CardContent>
            </Card>
          </div>
        )}
      </Draggable>
    );
  };
  
export default TaskCard;