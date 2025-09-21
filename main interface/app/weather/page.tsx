"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  ArrowLeft,
  Droplets,
  Wind,
  Gauge,
  Eye,
  MapPin,
} from "lucide-react";
import { useRouter } from "next/navigation";

interface WeatherApiResponse {
  location: {
    name: string;
    region: string;
  };
  current: {
    temp_c: number;
    humidity: number;
    wind_kph: number;
    wind_dir: string;
    pressure_mb: number;
    vis_km: number;
    uv: number;
    condition: {
      text: string;
      icon: string;
    };
  };
}

export default function WeatherPage() {
  const router = useRouter();
  const [location, setLocation] = useState("Fetching location...");
  const [weatherData, setWeatherData] = useState<WeatherApiResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          try {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
            const apiKey = process.env.NEXT_PUBLIC_WEATHER_API_KEY;
            const res = await fetch(
              `https://api.weatherapi.com/v1/current.json?key=${apiKey}&q=${lat},${lon}`
            );
            const data: WeatherApiResponse = await res.json();

            if (data?.location) {
              setLocation(`${data.location.name}, ${data.location.region}`);
            } else {
              setLocation("Location unavailable");
            }
            setWeatherData(data);
          } catch (error) {
            setErrorMsg("Failed to load weather data.");
            setLocation("Location unavailable");
          } finally {
            setLoading(false);
          }
        },
        () => {
          setErrorMsg("Geolocation permission denied.");
          setLocation("Location unavailable");
          setLoading(false);
        }
      );
    } else {
      setErrorMsg("Geolocation is not supported by this browser.");
      setLocation("Location unavailable");
      setLoading(false);
    }
  }, []);

  if (loading) {
    return <div>Loading weather data...</div>;
  }

  if (errorMsg) {
    return <div>Error: {errorMsg}</div>;
  }

  if (!weatherData) {
    return <div>No weather data available.</div>;
  }

  const uvIndex = weatherData?.current?.uv;
  const getUVLevel = (uv: number) => {
    if (uv <= 2) return { level: "Low", color: "text-green-600 bg-green-100" };
    if (uv <= 5) return { level: "Moderate", color: "text-yellow-600 bg-yellow-100" };
    if (uv <= 7) return { level: "High", color: "text-orange-600 bg-orange-100" };
    if (uv <= 10) return { level: "Very High", color: "text-red-600 bg-red-100" };
    return { level: "Extreme", color: "text-purple-600 bg-purple-100" };
  };

  const uvLevel = getUVLevel(uvIndex);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-card border-b border-border px-4 py-4">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={() => router.back()}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div>
            <h1 className="text-xl font-serif font-bold">Weather Forecast</h1>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <MapPin className="h-4 w-4" />
              <span>{location}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Page container wrapper */}
      <div className="max-w-4xl mx-auto px-4">
        {/* Current Weather */}
        <div className="max-w-lg mx-auto p-6 bg-card rounded-lg shadow-md">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                {weatherData?.current?.condition?.icon && (
                  <img
                    src={`https:${weatherData.current.condition.icon}`}
                    alt={weatherData.current.condition.text || "weather icon"}
                    width={32}
                    height={32}
                  />
                )}
                <span className="font-semibold text-lg">
                  Weather in {location}
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <div>
                  <p className="text-xl font-bold flex items-center gap-2">
                    <span>🌡️</span> {weatherData?.current?.temp_c ?? "--"} °C
                  </p>
                  <p className="text-muted-foreground mt-1">
                    {weatherData?.current?.condition?.text || "N/A"}
                  </p>
                </div>

                <div>
                  <p className="flex items-center gap-2 text-lg">
                    <Droplets className="h-5 w-5 text-primary" />
                    Humidity: {weatherData?.current?.humidity ?? "--"}%
                  </p>
                  <p className="flex items-center gap-2 text-lg mt-2">
                    <Wind className="h-5 w-5 text-primary" />
                    Wind: {weatherData?.current?.wind_kph ?? "--"} km/h {weatherData?.current?.wind_dir || "--"}
                  </p>
                </div>

                <div className="flex items-center gap-2 text-lg">
                  <Gauge className="h-5 w-5 text-primary" />
                  Pressure: {weatherData?.current?.pressure_mb ?? "--"} hPa
                </div>

                <div className="flex items-center gap-2 text-lg">
                  <Eye className="h-5 w-5 text-primary" />
                  Visibility: {weatherData?.current?.vis_km ?? "--"} km
                </div>
              </div>

              <div className="mt-6 flex items-center gap-2">
                <span className="text-sm text-muted-foreground">UV Index:</span>
                <Badge className={uvLevel.color}>
                  {uvIndex} - {uvLevel.level}
                </Badge>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Add other page sections here */}
      </div>
    </div>
  );
}
