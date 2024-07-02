import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Form } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

function App() {
  const [videoOn, setVideoOn] = useState(false);
  const [stats, setStats] = useState({ count: 0, history: [] });
  const [cameraIndex, setCameraIndex] = useState(0);
  const [availableCameras, setAvailableCameras] = useState([]);

  useEffect(() => {
    // Fetch available cameras when component mounts
    fetch('/api/get_available_cameras')
      .then(response => response.json())
      .then(data => {
        if (data.cameras && data.cameras.length > 0) {
          console.log("Available cameras: ", data.cameras);
          setAvailableCameras(data.cameras);
          setCameraIndex(data.cameras[0]);
        } else {
          console.warn("No cameras available");
          setAvailableCameras([]);
        }
      })
      .catch(error => console.error('Error: ', error));

    const interval = setInterval(() => {
      fetch('/api/get_stats')
        .then(response => response.json())
        .then(data => setStats(data));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const toggleVideo = () => {
    const action = videoOn ? 'stop_video' : 'start_video';
    fetch(`/api/${action}`)
      .then(response => response.json())
      .then(() => setVideoOn(!videoOn));
  };

  const setThreshMethod = (method) => {
    fetch(`/api/set_thresh_method/${method}`)
      .then(response => response.json())
      .then(data => console.log(data.status));
  };

  const handleCameraChange = (event) => {
    const newCameraIndex = parseInt(event.target.value);
    setCameraIndex(newCameraIndex);
    fetch(`/api/set_camera`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ camera_index: newCameraIndex }),
    })
      .then(response => response.json())
      .then(data => {
        if (data.status) {
          console.log(data.status);
        } else if (data.error) {
          console.error(data.error);
        }
      })
      .catch(error => console.error('Error: ', error));
  };


  return (
    <Container fluid>
      <Row className="mt-4">
        <Col md={8}>
          <Card>
            <Card.Body>
              <Card.Title>Video Feed</Card.Title>
              <img
                src="/api/video_feed"
                alt="Video feed"
                style={{ display: videoOn ? 'block' : 'none', width: '100%' }}
              />
            </Card.Body>
          </Card>
        </Col>
        <Col md={4}>
          <Card className="mb-4">
            <Card.Body>
              <Card.Title>Statistics</Card.Title>
              <p>Count: {stats.count}</p>
              <h6>Card History:</h6>
              <div className="card-history">
                {stats.history.map((card, index) => (
                  <div key={index}>{`${card[0]} of ${card[1]}`}</div>
                ))}
              </div>
            </Card.Body>
          </Card>
          <Card className="mb-4">
            <Card.Body>
              <Card.Title>Controls</Card.Title>
              <Button onClick={toggleVideo}>
                {videoOn ? 'Stop Video' : 'Start Video'}
              </Button>
            </Card.Body>
          </Card>
          <Card className='mb-4'>
            <Card.Body>
              <Card.Title>Thresholding Method</Card.Title>
              <Button onClick={() => setThreshMethod('original')} className="me-2 mb-2">Original</Button>
              <Button onClick={() => setThreshMethod('adaptive')} className="me-2 mb-2">Adaptive</Button>
              <Button onClick={() => setThreshMethod('otsu')} className="me-2 mb-2">Otsu</Button>
            </Card.Body>
          </Card>
          <Card className='mb-4'>
            <Card.Body>
              <Card.Title>Camera Selection</Card.Title>
              {availableCameras.length > 0 ? (
                <Form.Select
                  value={cameraIndex}
                  onChange={handleCameraChange}
                >
                  {availableCameras.map((camera, index) => (
                    <option key={index} value={index}>Camera {index}</option>
                  ))}
                </Form.Select>
              ) : (
                <p>No cameras available</p>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default App;